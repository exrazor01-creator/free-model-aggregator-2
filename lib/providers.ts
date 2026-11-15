// The trial brief: two JSON documents hidden in prose, and the catalog they make.
//
// A trial hands this entry a brief (the path in MAKEFUN_PROMPT_FILE). The brief is prose with two
// JSON documents in it, each carrying a `schema` string under this contest's prefix
// (`free-model-aggregator-02/`):
//
//   - the model list (`free-model-aggregator-02/models/v1`): one row per model, keyed by the id the
//     gateway routes (`openrouter/<vendor>/<model>:free`), with the model's published rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length) and its free flag.
//     Only a free model may ever serve a prompt; a paid row is never routed.
//   - the prompts (`free-model-aggregator-02/prompts/v1`): the prompts the house's client will send,
//     in arrival order, each with its id, its api (`openai` or `anthropic`), whether it streams, its
//     weight (1 or 2) and the requirements a model must meet to serve it. The entry routes each
//     arrival by its row, so the list is read before the first request.
//
// The brief may spell the lists `models`/`prompts` or the sibling contest's `ratings`/`manifest`,
// and one document may carry both lists; a document is found by the arrays it carries, never by a
// name alone. The prose around the documents is never parsed.
//
// Nothing here reads the clock or rolls dice: `node main.ts --list` and the register body build the
// same document, byte for byte, on a live pass and on its replay.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** The prefix every schema in this contest's brief carries. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The env var an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this entry reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** A string field, or '' when the value is not a string. */
function asText(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** A finite number field, or null when the value is not one. */
function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** A boolean field, or null when the value is not one. */
function asBool(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** One balanced JSON object found in the brief, with where it opens so documents stay in brief order. */
interface FoundDoc {
  start: number;
  value: Json;
}

/**
 * Every balanced JSON object in `text` that carries a string `schema`, in the order they open.
 *
 * One left-to-right pass keeps a stack of open braces; a closing brace completes the object that
 * opened at the matching `{`, string contents (with their escapes) are skipped, and a completed span
 * is parsed — a span that is not JSON on its own is dropped and the scan goes on. Nested objects are
 * found too, so the result is ordered by opening brace before it is used.
 */
function scanDocuments(text: string): FoundDoc[] {
  const found: FoundDoc[] = [];
  const stack: number[] = [];
  let inString = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') stack.push(i);
    else if (ch === '}' && stack.length > 0) {
      const start = stack.pop() as number;
      try {
        const parsed: unknown = JSON.parse(text.slice(start, i + 1));
        if (isRecord(parsed) && typeof parsed.schema === 'string') found.push({ start, value: parsed });
      } catch {
        // Not JSON on its own: look on.
      }
    }
  }
  found.sort((a, b) => a.start - b.start);
  return found;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array, and the key
 * it was found under. A brief that spells its lists in the sibling contest's names still reads.
 */
function findListDocument(docs: readonly FoundDoc[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const { value: doc } of docs) {
    if (!asText(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row
 * that gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the
 * gateway spells it (`openrouter/vendor/model:free`). The brief lists free models, so a row is free
 * unless it says `free: false`; a paid row is never routed.
 */
function rowToModel(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const rawId = asText(value.id);
  const raw = rawId !== '' ? rawId : asText(value.model);
  if (raw === '') return null;
  const providerField = asText(value.provider);
  const named = providerField !== '' ? providerField : asText(value.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: asText(value.display_name) !== '' ? asText(value.display_name) : raw,
    rating_milli: asNumber(value.rating_milli) ?? 0,
    context_length: asNumber(value.context_length),
    tools: asBool(value.tools),
    vision: asBool(value.vision),
    json: asBool(value.json),
    free: value.free !== false,
  };
}

/** One prompt row: only the requirements the brief marks true are kept, so absence means "not needed". */
function rowToPrompt(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = asText(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const minContext = asNumber(requires.min_context);
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(minContext !== null ? { min_context: minContext } : {}),
    },
  };
}

/** The rows of one list document, converted, with unusable rows dropped. */
function collectRows<T>(doc: Json, key: string, convert: (value: unknown) => T | null): T[] {
  const rows = doc[key] as unknown[];
  const out: T[] = [];
  for (const row of rows) {
    const converted = convert(row);
    if (converted !== null) out.push(converted);
  }
  return out;
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = scanDocuments(text);
  const modelsDoc = findListDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = collectRows(modelsDoc.doc, modelsDoc.key, rowToModel);
  const promptsDoc = findListDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  return { models, prompts: collectRows(promptsDoc.doc, promptsDoc.key, rowToPrompt), error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local address: the
 * gateway's origin differs between a live pass and its replay, and the catalog is part of the state
 * export a replay must reproduce byte for byte.
 *
 * This entry lists every free model in the brief — exactly the fourteen dialable ids the catalog
 * column scores — so keep it that way: register this set and keep GET /v1/models (both auth
 * headers) exactly that set, since listing fewer loses the column as surely as listing more.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const byId = new Map<string, CatalogModel>();
  for (const model of brief.models) {
    if (!model.free || byId.has(model.id)) continue;
    byId.set(model.id, { id: model.id, free: true, context_length: model.context_length });
  }
  const models = [...byId.values()];
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
