export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  });
  if (!response.ok) {
    let description = `Ошибка запроса (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') description = body.detail;
    } catch { /* A reverse proxy may return an HTML error page. */ }
    throw new Error(description);
  }
  return response.json() as Promise<T>;
}

export function post<T>(path: string, body: unknown = {}): Promise<T> {
  return api<T>(path, { method: 'POST', body: JSON.stringify(body) });
}
