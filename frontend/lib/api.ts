/**
 * Centralized API client for the XAI NIDS FastAPI backend.
 *
 * Backend contract (see app/api/schemas.py):
 *   GET    /              -> { name, status }
 *   GET    /health        -> { status, artifacts_loaded }
 *   GET    /model-info    -> { model, full_model, lightweight_model, top_features }
 *   POST   /predict       -> SinglePredictResponse
 *   POST   /predict/batch -> BatchPredictResponse
 */

export type ModelName = "full" | "light";

export interface HealthResponse {
  status: "healthy" | "degraded";
  artifacts_loaded: boolean;
}

export interface ModelSubInfo {
  features: number;
  accuracy: number;
}

export interface ModelInfoResponse {
  model: string;
  full_model: ModelSubInfo;
  lightweight_model: ModelSubInfo;
  top_features: string[];
}

export interface ShapItem {
  feature: string;
  shap_value: number;
}

export interface SinglePredictResponse {
  prediction: number;
  label: string;
  probability: number;
  probabilities: Record<string, number>;
  model: ModelName;
  explanation: ShapItem[];
}

export interface BatchRow {
  prediction: number;
  label: string;
  probability: number;
}

export interface BatchPredictResponse {
  model: ModelName;
  total_rows: number;
  normal: number;
  attack: number;
  results: BatchRow[];
}

export interface ApiError {
  status: number;
  message: string;
}

function baseUrl(): string {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not set. Copy frontend/.env.local.example to .env.local.",
    );
  }
  return url.replace(/\/$/, "");
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body?.detail)) {
        // FastAPI/Pydantic 422 validation error array → first message
        const first = body.detail[0];
        if (first && typeof first.msg === "string") detail = first.msg;
      }
    } catch {
      /* ignore non-JSON bodies */
    }
    const err: ApiError = { status: res.status, message: detail };
    throw err;
  }
  return res.json() as Promise<T>;
}

export async function request<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    // Network-level failures (DNS, connection refused, offline) throw TypeError
    // before reaching `handle()`. Convert into a friendly ApiError.
    if (
      e instanceof TypeError &&
      /fetch|network|failed/i.test(e.message)
    ) {
      throw {
        status: 0,
        message: "Unable to connect to the analysis server. Please ensure the backend is running.",
      } satisfies ApiError;
    }
    throw e;
  }
}

export async function getHealth(): Promise<HealthResponse> {
  return request(async () => {
    const res = await fetch(`${baseUrl()}/health`, { cache: "no-store" });
    return handle<HealthResponse>(res);
  });
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return request(async () => {
    const res = await fetch(`${baseUrl()}/model-info`, { cache: "no-store" });
    return handle<ModelInfoResponse>(res);
  });
}

export async function predictSingle(
  features: Record<string, number>,
  model: ModelName = "full",
): Promise<SinglePredictResponse> {
  return request(async () => {
    const res = await fetch(`${baseUrl()}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, features }),
    });
    return handle<SinglePredictResponse>(res);
  });
}

export async function predictBatch(
  file: File,
  model: ModelName = "full",
): Promise<BatchPredictResponse> {
  return request(async () => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${baseUrl()}/predict/batch?model=${model}`, {
      method: "POST",
      body: fd,
    });
    return handle<BatchPredictResponse>(res);
  });
}

/**
 * The 39 raw feature names the backend preprocessor expects.
 * Kept here as the single source of truth for the UI form.
 */
export const RAW_FEATURES: string[] = [
  "dur",
  "spkts",
  "dpkts",
  "sbytes",
  "dbytes",
  "rate",
  "sttl",
  "dttl",
  "sload",
  "dload",
  "sloss",
  "dloss",
  "sinpkt",
  "dinpkt",
  "sjit",
  "djit",
  "swin",
  "stcpb",
  "dtcpb",
  "dwin",
  "tcprtt",
  "synack",
  "ackdat",
  "smean",
  "dmean",
  "trans_depth",
  "response_body_len",
  "ct_srv_src",
  "ct_state_ttl",
  "ct_dst_ltm",
  "ct_src_dport_ltm",
  "ct_dst_sport_ltm",
  "ct_dst_src_ltm",
  "is_ftp_login",
  "ct_ftp_cmd",
  "ct_flw_http_mthd",
  "ct_src_ltm",
  "ct_srv_dst",
  "is_sm_ips_ports",
];

/**
 * Map post-transform Top-K feature names back to their raw column.
 * Example: "num__sttl" -> "sttl".
 *
 * NOTE: The deployed `xgb_light` model is a slice of the 39-feature
 * transformed matrix (see `app/train.py`), so even when the user picks the
 * "lightweight" option they must still submit ALL 39 raw features. The
 * SHAP response will only contain the 10 top contributors, which is what
 * the UI highlights.
 */
export const TOP10_RAW: string[] = [
  "sttl",
  "ct_dst_src_ltm",
  "ct_dst_sport_ltm",
  "sbytes",
  "smean",
  "ct_srv_dst",
  "ct_srv_src",
  "ct_dst_ltm",
  "dbytes",
  "dmean",
];

/** A demo sample derived from the project's real training CSV. */
export const DEMO_SAMPLE: Record<string, number> = {
  dur: 0.000011,
  spkts: 2,
  dpkts: 2,
  sbytes: 124,
  dbytes: 174,
  rate: 90909.0909,
  sttl: 254,
  dttl: 252,
  sload: 5454545.45,
  dload: 7695652.17,
  sloss: 0,
  dloss: 0,
  sinpkt: 0.011,
  dinpkt: 0.011,
  sjit: 0,
  djit: 0,
  swin: 0,
  stcpb: 65535,
  dtcpb: 65535,
  dwin: 0,
  tcprtt: 0.022,
  synack: 0.011,
  ackdat: 0.01,
  smean: 62,
  dmean: 87,
  trans_depth: 1,
  response_body_len: 174,
  ct_srv_src: 1,
  ct_state_ttl: 1,
  ct_dst_ltm: 1,
  ct_src_dport_ltm: 1,
  ct_dst_sport_ltm: 1,
  ct_dst_src_ltm: 1,
  is_ftp_login: 0,
  ct_ftp_cmd: 0,
  ct_flw_http_mthd: 0,
  ct_src_ltm: 2,
  ct_srv_dst: 1,
  is_sm_ips_ports: 0,
};
