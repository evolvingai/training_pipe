import crypto from 'node:crypto';

function safeEqual(a, b) {
  const left = Buffer.from(String(a));
  const right = Buffer.from(String(b));
  return left.length === right.length && crypto.timingSafeEqual(left, right);
}

function cookies(req) {
  return Object.fromEntries(
    String(req.headers.cookie || '')
      .split(';')
      .map((part) => part.trim().split('='))
      .filter(([key, value]) => key && value)
      .map(([key, value]) => [key, decodeURIComponent(value)]),
  );
}

function signature(value, secret) {
  return crypto.createHmac('sha256', secret).update(value).digest('base64url');
}

export function makeSession(username, secret, now = Date.now()) {
  const payload = Buffer.from(JSON.stringify({ username, expiresAt: now + 12 * 60 * 60 * 1000 })).toString('base64url');
  return `${payload}.${signature(payload, secret)}`;
}

export function sessionUser(req, secret) {
  const jar = cookies(req);
  const token = jar.trainer_session || '';
  const split = token.lastIndexOf('.');
  if (split < 1) return null;
  const payload = token.slice(0, split);
  if (!safeEqual(token.slice(split + 1), signature(payload, secret))) return null;
  try {
    const parsed = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8'));
    if (!parsed.username || parsed.expiresAt < Date.now()) return null;
    return String(parsed.username);
  } catch {
    return null;
  }
}

export function validLogin(username, password, expectedUser, expectedPassword) {
  return safeEqual(username, expectedUser) && safeEqual(password, expectedPassword);
}

export function validAgent(req, token) {
  const header = String(req.headers.authorization || '');
  return header.startsWith('Bearer ') && safeEqual(header.slice(7), token);
}

export function sessionCookie(value, secure = false) {
  const attrs = ['Path=/', 'HttpOnly', 'SameSite=Strict', 'Max-Age=43200'];
  if (secure) attrs.push('Secure');
  return `trainer_session=${encodeURIComponent(value)}; ${attrs.join('; ')}`;
}

export function clearSessionCookie(secure = false) {
  const attrs = ['Path=/', 'HttpOnly', 'SameSite=Strict', 'Max-Age=0'];
  if (secure) attrs.push('Secure');
  return `trainer_session=; ${attrs.join('; ')}`;
}
