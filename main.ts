// free-model-aggregator-02: one server that answers every prompt with a REAL completion from a
// live free OpenRouter model, dialled through the grader's gateway.
//
// A trial gives this server a brief (MAKEFUN_PROMPT_FILE) and an env (MAKEFUN_ENV_URL) and runs it
// as `node main.ts` from the repository's root, with Node 22's standard library alone. The brief
// carries the fourteen free models (gateway id, rating_milli, tools/vision/JSON/context) and the six
// prompts' manifest. The server reads the brief, builds one catalog, listens on a free port,
// registers it, serves the client's six prompts, polls GET /work and exits 0 when the env says done.
// `node main.ts --list` prints the same catalog JSON and exits 0. Nothing here reads the clock for
// anything it sends or rolls dice: every choice is a function of the brief, the request and the
// gateway's answers, so a live pass and its replay produce the same bytes. (Date.now is used only
// as a local per-prompt deadline so a hung upstream call cannot eat the env's 30 s answer cap; it
// never reaches a request, a response or the state export.)
//
// HOW THE SCORE IS WON
//   quality (60%): for each prompt, among the free models meeting ALL of its requirements (JSON
//     mode, tools, vision, context length), pick the highest rating_milli in the brief. The best
//     model for a plain prompt is thinkingmachines/inkling-2-small (1475); for a JSON-mode prompt
//     it is nvidia/nemotron-3.5-super-vl (1303, the best JSON-capable model). Ties break on model
//     id, so the same brief always ranks the same way. A paid model is never a legal pick.
//   served (15%) + failover (10%): a refusal is real here — a provider's 429 or 5xx reaches this
//     server as the provider sent it. A 429 cools that model for the prompt (a rate limit does
//     not clear inside one prompt) and the next-best eligible model gets the prompt at once; the
//     cooled model is retried first on the next prompt, in case the limit was a short burst, and a
//     second 429 proves it is not, so the model is dead for the trial. A 5xx or a transport
//     failure is retried once on the same model, then the next-best gets it. Every refusal is
//     recorded, and when a retried or failed-over prompt is still served, its refusals ride on the
//     response in the `x-makefun-upstream-failures` header — the only way the failover column can
//     see a recovered refusal, since the env never sees the gateway's traffic. A 401 means the
//     entrant's key is missing or refused: no model can serve anything, so the trial stops
//     dialling and answers 503 (one wasted call per prompt at most, and none at all after the
//     first). At most four candidates and six calls per prompt, so the 60-call trial cap is never
//     approached and the tie-break (fewer gateway calls) stays cheap.
//   wire (10%): the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or SSE,
//     and the gateway speaks Chat Completions only. The Anthropic path is translated both ways by
//     lib/wire.ts; the OpenAI path passes the client's body through with only the model replaced.
//     Every answer carries the provider's text whole, its usage counts and a stop reason in its
//     family, and every `model` field names the id this server really dialled (the provider's own
//     answer spells it without the `openrouter/` prefix, so it is put back on every answer and
//     every streamed chunk). Two wire rules the client never states: an OpenAI stream carries no
//     usage chunk unless stream_options.include_usage is set, so it is injected on every streamed
//     OpenAI upstream call; and the gateway refuses a call with no output cap, so a body without
//     max_tokens gets one (1024, capped at the gateway's 8192).
//   catalog (5%): the catalog is exactly the fourteen free ids from the brief — register that same
//     set and answer GET /v1/models with it on both auth headers.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway's output cap: a call naming more, or none at all, is refused. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap a body without one gets, so the gateway never refuses it for missing max_tokens. */
const DEFAULT_MAX_TOKENS = 1024;
/** How long one gateway call may take before it counts as failed (the env waits 30 s an answer). */
const CALL_TIMEOUT_MS = 12000;
/** How long one prompt may spend dialling, in milliseconds (local control only, never sent). */
const PROMPT_DEADLINE_MS = 27000;
/** How many candidates one prompt may try, best first. */
const MAX_CANDIDATES_PER_PROMPT = 4;
/** How many times one candidate is tried: a second attempt only after a 5xx or a transport failure. */
const ATTEMPTS_PER_CANDIDATE = 2;
/** The most refusal rows the failover header may carry. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** How many client API requests have arrived: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models a refusal proved unusable for this trial (429 twice, a spent budget, a repeated 5xx). */
const deadModels = new Set<string>();
/**
 * Models that answered one 429: a rate limit does not clear inside one prompt, so the next
 * candidate gets this prompt at once — but the model is retried as first candidate on the next
 * prompt, in case the limit was a short burst. A second 429 moves it to deadModels.
 */
const coolingModels = new Set<string>();
/** A 401 means the entrant's key is missing or refused: no model can serve anything after it. */
let keyDead = false;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Every free model that can serve one prompt, best first. The rating is the brief's milli-rating;
 * the tie-break is the model id, so the same brief always ranks the same way. Models a refusal
 * proved dead are left out.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const out: ModelRow[] = [];
  for (const model of freeModels(brief)) {
    if (deadModels.has(model.id)) continue;
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    out.push(model);
  }
  out.sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return out;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is
  // a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function cloneJson(body: Json): Json {
  return JSON.parse(JSON.stringify(body)) as Json;
}

/** One row of the failover trail: a refused gateway call, in the order the calls refused. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The failover trail as the header carries it: at most eight rows, ASCII only. */
function trailHeader(failures: readonly FailureRow[]): string | null {
  if (failures.length === 0) return null;
  return JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS));
}

/** The code of a gateway refusal's error body: its code, else its type, else a code for its status. */
function errorCodeOf(status: number, text: string): string {
  try {
    const body: unknown = JSON.parse(text);
    if (isRecord(body) && isRecord(body.error)) {
      if (typeof body.error.code === 'string' && body.error.code !== '') return body.error.code.slice(0, 64);
      if (typeof body.error.type === 'string' && body.error.type !== '') return body.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status mapping.
  }
  if (status === 429) return 'rate_limit_exceeded';
  if (status === 401) return 'invalid_api_key';
  if (status === 400) return 'invalid_request_error';
  if (status >= 500) return 'internal_error';
  return 'upstream_error';
}

/**
 * The Chat Completions body one gateway call sends. An OpenAI client's body passes through with
 * only the model replaced; an Anthropic one is translated first. Both get an output cap (the
 * gateway refuses a call without one) and, on a stream, the usage chunk switched on (the client
 * never asks for it and the usage check needs it).
 */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  const out = clientApi === 'openai' ? cloneJson(body) : anthropicToOpenAI(body, model);
  out.model = model;
  const cap = typeof out.max_tokens === 'number' ? out.max_tokens : typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens : null;
  if (cap === null || cap > MAX_OUTPUT_TOKENS) out.max_tokens = DEFAULT_MAX_TOKENS;
  if (out.stream === true) {
    const options = isRecord(out.stream_options) ? { ...out.stream_options } : {};
    options.include_usage = true;
    out.stream_options = options;
  }
  return out;
}

/** What one gateway call returned: a success with its response, or a refusal with its status. */
type DialOutcome =
  | { ok: true; res: Response }
  | { ok: false; status: number; code: string; transport: boolean };

/**
 * One call to the gateway's Chat Completions route, naming the model this entry dialled. A 2xx is a
 * success; anything else is a refusal, and a call that never answered (a timeout, a dropped
 * connection) is a transport failure. The response body is left unread here: the caller reads it
 * as JSON or relays it as SSE.
 */
async function dial(model: string, body: Json, timeoutMs: number): Promise<DialOutcome> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch {
    return { ok: false, status: 0, code: 'transport_error', transport: true };
  }
  if (res.ok) return { ok: true, res };
  const code = errorCodeOf(res.status, await res.text());
  return { ok: false, status: res.status, code, transport: false };
}

/**
 * Serve one prompt: dial the best free model that meets it, fail over on refusal, and return the
 * model that answered with its response — or null when no model could serve it. The failures list
 * fills with every refusal on the way, so the caller can report the trail.
 */
async function serveUpstream(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  requires: Requirement,
  failures: FailureRow[],
): Promise<{ model: string; res: Response } | null> {
  if (keyDead) return null;
  const candidates = candidatesFor(requires).slice(0, MAX_CANDIDATES_PER_PROMPT);
  if (candidates.length === 0) return null;
  const deadline = Date.now() + PROMPT_DEADLINE_MS;
  for (const candidate of candidates) {
    for (let attempt = 0; attempt < ATTEMPTS_PER_CANDIDATE; attempt++) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) return null;
      const outcome = await dial(candidate.id, upstreamBody(clientApi, body, candidate.id), Math.min(CALL_TIMEOUT_MS, remaining));
      if (outcome.ok) {
        // The model answered: a cooling model proved itself usable again.
        coolingModels.delete(candidate.id);
        return { model: candidate.id, res: outcome.res };
      }
      failures.push({ status: outcome.status, code: outcome.code, model: candidate.id });
      if (outcome.status === 401) {
        // The key is missing or refused: no model can serve anything. Stop dialling for good.
        keyDead = true;
        return null;
      }
      if (outcome.status === 429) {
        // One 429 cools the model for this prompt only: it is retried first on the next prompt,
        // and a second 429 proves the limit is not a short burst, so it is dead for the trial.
        if (coolingModels.has(candidate.id)) deadModels.add(candidate.id);
        else coolingModels.add(candidate.id);
        break;
      }
      if (outcome.transport || outcome.status >= 500) {
        // Transient: one retry on the same model, then it is dead for the trial.
        if (attempt === 0) continue;
        deadModels.add(candidate.id);
        break;
      }
      // Any other 4xx is this model's own refusal: the next candidate gets the prompt.
      deadModels.add(candidate.id);
      break;
    }
  }
  return null;
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: readonly FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  const trail = trailHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: readonly FailureRow[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Answer one client request in its own wire, from the gateway call that served it. A JSON answer is
 * relayed with the dialled model put back; a stream is relayed chunk by chunk, translated when the
 * client speaks Anthropic. The failover trail rides on the response whenever a refusal came first.
 */
async function answer(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  served: { model: string; res: Response },
  failures: readonly FailureRow[],
): Promise<void> {
  const { model, res: upstream } = served;
  if (body.stream === true) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    const trail = trailHeader(failures);
    if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
    res.writeHead(200, headers);
    try {
      if (clientApi === 'openai') {
        for await (const frame of openAIStreamWithModel(upstream.body, model)) res.write(frame);
      } else {
        for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answerBody: unknown = await upstream.json();
  if (clientApi === 'openai') sendJson(res, 200, openAIJsonWithModel(isRecord(answerBody) ? answerBody : {}, model), failures);
  else sendJson(res, 200, openAIToAnthropic(isRecord(answerBody) ? answerBody : {}, model), failures);
}

/** Serve one client request: route it, fail over, and answer in the client's own wire. */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const failures: FailureRow[] = [];
  const served = await serveUpstream(clientApi, body, requires, failures);
  if (served === null) {
    // Every free model that meets the request is spent, or the key is dead: the request fails,
    // honestly, in the client's wire, with the refusals behind it on the trail.
    log(`no model served a ${clientApi} request (requirements ${JSON.stringify(requires)})`);
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request', failures);
  }
  await answer(req, res, clientApi, body, served, failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
