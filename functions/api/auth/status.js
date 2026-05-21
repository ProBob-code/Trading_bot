/**
 * GET /api/auth/status
 * Check if the user is authenticated via session cookie.
 */
export async function onRequestGet(context) {
    const { env, request } = context;

    try {
        const db = env.trading_bot_v2;
        if (!db) {
            return jsonResponse({ authenticated: false });
        }

        // Read session token from cookie
        const cookie = request.headers.get("Cookie") || "";
        const token = getCookie(cookie, "goatbot_session");

        if (!token) {
            return jsonResponse({ authenticated: false });
        }

        // Look up session in D1
        const session = await db.prepare(
            "SELECT s.user_id, s.expires_at, u.username, u.is_verified FROM auth_sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ?"
        ).bind(token).first();

        if (!session) {
            return jsonResponse({ authenticated: false }, clearSessionCookie());
        }

        // Check expiry
        if (new Date(session.expires_at) < new Date()) {
            // Session expired — clean up
            await db.prepare("DELETE FROM auth_sessions WHERE token = ?").bind(token).run();
            return jsonResponse({ authenticated: false }, clearSessionCookie());
        }

        return jsonResponse({
            authenticated: true,
            user: {
                id: session.user_id,
                username: session.username,
                is_verified: session.is_verified
            }
        });

    } catch (err) {
        console.error("Auth status error:", err);
        return jsonResponse({ authenticated: false });
    }
}

// ── Helpers ──

function getCookie(cookieHeader, name) {
    const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${name}=([^;]*)`));
    return match ? match[1] : null;
}

function jsonResponse(data, extraHeaders = {}) {
    return new Response(JSON.stringify(data), {
        headers: {
            "Content-Type": "application/json",
            ...extraHeaders
        }
    });
}

function clearSessionCookie() {
    return {
        "Set-Cookie": "goatbot_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax"
    };
}
