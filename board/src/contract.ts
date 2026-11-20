// Wire contract between the collector (or mock/replay) and the board.
// Every message has `type` and usually `t_ns` (wall-clock ns at the source).
// Fields may be missing or null; the board must degrade to "unstained", never crash.

export interface Coverage {
  requests: boolean;
  steps: boolean;
  kv: boolean;
  spec: boolean;
  gpu: boolean;
  moe?: boolean; // expert routing (moe.tick) is reported
}

export interface Hello {
  type: 'hello';
  t_ns?: number;
  engine?: string | null;
  engine_version?: string | null;
  model?: string | null;
  gpus?: number | null;
  gpu_ids?: number[] | null;
  coverage?: Partial<Coverage> | null;
  config?: Record<string, unknown> | null;
  label?: string | null; // free-form caption override (mock/replay)
  instances?: Record<string, InstanceInfo> | null;
  replay?: { file?: string; speed?: number; recorded_utc?: string } | null; // set by io-replay-ws
}

export interface Usage {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  prompt_tokens_details?: { cached_tokens?: number | null } | null;
}

export interface EngineTick {
  type: 'engine.tick';
  t_ns?: number;
  engine?: string;
  source?: 'tap' | 'metrics' | string;
  running?: number | null;
  waiting?: number | null;
  kv_usage?: number | null;
  prefix_hit_rate?: number | null;
  prompt_tok_s?: number | null;
  gen_tok_s?: number | null;
  steps_per_s?: number | null;
  preemptions_total?: number | null;
  active_requests?: number | null;
  spec?: { accept_rate?: number | null; mean_accept_len?: number | null } | null;
  instances?: Record<string, InstanceInfo> | null;
}

export interface GpuSample {
  i: number;
  util?: number | null;
  mem_util?: number | null;
  mem_used_gb?: number | null;
  mem_total_gb?: number | null;
  power_w?: number | null;
  sm_clock_mhz?: number | null;
}

export interface GpuTick {
  type: 'gpu.tick';
  t_ns?: number;
  gpus?: GpuSample[] | null;
}

export interface FinishedReq {
  rid?: string | null;
  request_id?: string | null;
  queued_s?: number | null;
  prefill_s?: number | null;
  decode_s?: number | null;
  prompt_tokens?: number | null;
  cached_tokens?: number | null;
  gen_tokens?: number | null;
  preemptions?: number | null;
}

export interface EngineStep {
  type: 'engine.step';
  t_ns?: number;
  step?: number;
  running?: number | null;
  waiting?: number | null;
  kv_usage?: number | null;
  // generic shape
  prefill_reqs?: number | null;
  prefill_tokens?: number | null;
  decode_reqs?: number | null;
  decode_tokens?: number | null;
  dur_ms?: number | null;
  // vLLM tap shape
  iter?: { ctx_reqs?: number; ctx_tokens?: number; gen_reqs?: number; gen_tokens?: number; elapsed_ms?: number; dummy?: boolean } | null;
  gen_tokens?: number | null;
  preempted?: number | null;
  prompt?: { computed?: number; cached?: number; external?: number; total?: number } | null;
  prefix?: { queries?: number; hits?: number } | null;
  spec?: { drafts?: number; draft_tokens?: number; accepted_tokens?: number; accepted_per_pos?: number[] } | null;
  finished?: FinishedReq[] | null;
}

export interface AdvisorNote {
  type: 'advisor.note';
  t_ns?: number;
  id: string;
  agent?: string | null;
  title?: string | null;
  body?: string | null;
  predicted?: string | null;
  measured?: string | null;
  region?: 'membrane' | 'queue' | 'kv' | 'decode' | 'compute' | string | null;
  status?: 'info' | 'proposed' | 'testing' | 'measured' | 'confirmed' | 'rejected' | 'neutral' | 'failed' | 'tested' | 'cleared' | string | null;
  sample?: boolean; // mock-only marker
  kind?: 'experiment' | string | null;
  result?: Record<string, BenchResult> | null;
  deltas?: Record<string, { ttft_p90?: number | null; e2e_p90?: number | null; tok_s?: number | null; [k: string]: number | null | undefined }> | null;
  baseline?: string | null;
  goal?: string | null;
  speed?: number | null;
  config?: ExpConfig | null;
  experiment?: { args?: string[] | null; bench?: string[] | null; goal?: string | null } | null;
  experiment_id?: string | null;
}

export interface BenchResult {
  ttft_p50_ms?: number | null;
  ttft_p90_ms?: number | null;
  e2e_p90_ms?: number | null;
  tok_s?: number | null;
  errors?: number | null;
}

export interface ExpConfig {
  engine?: string | null;
  model?: string | null;
  gpus?: string | number | null;
  tp?: number | null;
  args?: string[] | null;
  pd_prefill?: string | null;
  pd_decode?: string | null;
}

export interface InstanceInfo {
  role?: 'prefill' | 'decode' | 'engine' | 'replica' | string | null;
  url?: string | null; // worker URL, as carried by proxy events (req.route / prefill_start / decode_start)
  running?: number | null;
  waiting?: number | null;
  kv_usage?: number | null;
  steps_per_s?: number | null;
  gen_tok_s?: number | null;
  prompt_tok_s?: number | null;
  config?: Record<string, unknown> | null;
}

// Loose message type: the board reads fields defensively.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Msg = { type: string; t_ns?: number; [k: string]: any };

export interface Source {
  readonly label: string;
  start(emit: (m: Msg) => void, status: (s: string) => void): void;
  stop(): void;
}
