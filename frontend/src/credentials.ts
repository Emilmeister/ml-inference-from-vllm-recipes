import type { CredentialInput } from "./types";

export type { CredentialInput } from "./types";

export const STORAGE_KEY = "vllm-cloudru.credentials.v1";
export const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

export type StoredCredentials = CredentialInput & { expiresAt: number };

export function saveCredentials(
  input: CredentialInput,
  now = Date.now(),
): StoredCredentials {
  const stored: StoredCredentials = {
    projectId: input.projectId,
    keyId: input.keyId,
    keySecret: input.keySecret,
    ...(input.hfSecretId ? { hfSecretId: input.hfSecretId } : {}),
    expiresAt: now + SEVEN_DAYS_MS,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
  return stored;
}

export function loadCredentials(now = Date.now()): StoredCredentials | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const value: unknown = JSON.parse(raw);
    if (!isStoredCredentials(value) || value.expiresAt <= now) {
      clearCredentials();
      return null;
    }
    return value;
  } catch {
    clearCredentials();
    return null;
  }
}

export function clearCredentials(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Storage can be denied by browser privacy settings; there is nothing else to clear.
  }
}

function isStoredCredentials(value: unknown): value is StoredCredentials {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.projectId === "string" &&
    candidate.projectId.length > 0 &&
    typeof candidate.keyId === "string" &&
    candidate.keyId.length > 0 &&
    typeof candidate.keySecret === "string" &&
    candidate.keySecret.length > 0 &&
    (candidate.hfSecretId === undefined || typeof candidate.hfSecretId === "string") &&
    typeof candidate.expiresAt === "number" &&
    Number.isFinite(candidate.expiresAt)
  );
}
