/**
 * POST /api/auth/logout
 * Destroy session and clear cookie.
 */
export async function onRequestPost(context) {
    const { env, request } = context;

    try {
        const db = env.trading_bot_v2;

        // Read session token from cookie
        const cookie = request.headers.get("Cookie") || "";
        const token = getCookie(cookie, "goatbot_session");

        // Delete session from D1 if it exists
        if (token && db) {
            await db.prepare("DELETE FROM auth_sessions WHERE token = ?").bind(token).run();
        }

        return new Response(JSON.stringify({ success: true, message: "Logged out" }), {
            headers: {
                "Content-Type": "application/json",
                "Set-Cookie": "goatbot_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax"
            }
        });

    } catch (err) {
        console.error("Logout error:", err);
        // Still clear the cookie even if DB delete fails
        return new Response(JSON.stringify({ success: true, message: "Logged out" }), {
            headers: {
                "Content-Type": "application/json",
                "Set-Cookie": "goatbot_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax"
            }
        });
    }
}

function getCookie(cookieHeader, name) {
    const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${name}=([^;]*)`));
    return match ? match[1] : null;
}
