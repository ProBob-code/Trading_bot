/**
 * POST /api/auth/login
 * Authenticate with username + password, create session.
 */
export async function onRequestPost(context) {
    const { env, request } = context;

    try {
        const db = env.trading_bot_v2;
        if (!db) {
            return jsonResponse({ success: false, error: "Database not available" }, 500);
        }

        const { username, password } = await request.json();

        if (!username || !password) {
            return jsonResponse({ success: false, error: "Username and password are required" }, 400);
        }

        // Look up user (plain-text password check — matches existing Flask backend behavior)
        const user = await db.prepare(
            "SELECT id, username, password_hash, is_verified FROM users WHERE username = ?"
        ).bind(username).first();

        if (!user || user.password_hash !== password) {
            return jsonResponse({ success: false, error: "Invalid credentials" }, 401);
        }

        // Create session
        const token = generateToken();
        const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(); // 7 days

        await db.prepare(
            "INSERT INTO auth_sessions (token, user_id, expires_at) VALUES (?, ?, ?)"
        ).bind(token, user.id, expiresAt).run();

        return new Response(JSON.stringify({ success: true, message: "Logged in successfully" }), {
            headers: {
                "Content-Type": "application/json",
                "Set-Cookie": `goatbot_session=${token}; Path=/; Max-Age=${7 * 24 * 60 * 60}; HttpOnly; Secure; SameSite=Lax`
            }
        });

    } catch (err) {
        console.error("Login error:", err);
        return jsonResponse({ success: false, error: "Server error" }, 500);
    }
}

// ── Helpers ──

function generateToken() {
    const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let token = "";
    const array = new Uint8Array(48);
    crypto.getRandomValues(array);
    for (const byte of array) {
        token += chars[byte % chars.length];
    }
    return token;
}

function jsonResponse(data, status = 200) {
    return new Response(JSON.stringify(data), {
        status,
        headers: { "Content-Type": "application/json" }
    });
}
