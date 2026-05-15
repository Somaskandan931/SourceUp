export const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const authHeader = () => {
  const t = localStorage.getItem('su_token');
  return t ? { Authorization: `Bearer ${t}` } : {};
};

const parseResponse = async (response) => {
  const data = await response.json().catch(() => ({}));
  return response.ok ? data : { error: true, detail: data.detail || 'Request failed' };
};

const networkError = (error) => ({
  error: true,
  detail: error?.message === 'Failed to fetch'
    ? `Could not reach the backend at ${API}. Start FastAPI on port 8000 and try again.`
    : error?.message || 'Network error',
});

export const post = (url, body) =>
  fetch(`${API}${url}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
  }).then(parseResponse).catch(networkError);

export const get = (url) =>
  fetch(`${API}${url}`, { headers: authHeader() }).then(parseResponse).catch(networkError);
