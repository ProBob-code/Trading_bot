/**
 * POST /api/auth/verify
 * Verify OTP, set username/password, auto-login.
 */
export async function onRequestPost(context) {
    const { env, request } = context;

    try {
        const db = env.trading_bot_v2;
        if (!db) {
            return jsonResponse({ success: false, error: "Database not available" }, 500);
        }

        const { mobile, otp, username, password } = await request.json();

        if (!mobile || !otp || !username || !password) {
            return jsonResponse({ success: false, error: "All fields are required" }, 400);
        }

        // Look up user by mobile
        const user = await db.prepare(
            "SELECT id, otp FROM users WHERE mobile = ?"
        ).bind(mobile).first();

        if (!user) {
            return jsonResponse({ success: false, error: "User not found" }, 404);
        }

        if (user.otp !== otp) {
            return jsonResponse({ success: false, error: "Invalid OTP" }, 401);
        }

        // Check if username is already taken
        const existingUser = await db.prepare(
            "SELECT id FROM users WHERE username = ? AND id != ?"
        ).bind(username, user.id).first();

        if (existingUser) {
            return jsonResponse({ success: false, error: "Username already taken" }, 409);
        }

        // Verify user — set username, password, mark as verified
        await db.prepare(
            "UPDATE users SET username = ?, password_hash = ?, is_verified = 1, otp = NULL WHERE id = ?"
        ).bind(username, password, user.id).run();

        // Auto-login: Create session
        const token = generateToken();
        const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();

        await db.prepare(
            "INSERT INTO auth_sessions (token, user_id, expires_at) VALUES (?, ?, ?)"
        ).bind(token, user.id, expiresAt).run();

        return new Response(JSON.stringify({
            success: true,
            message: "Account verified and logged in"
        }), {
            headers: {
                "Content-Type": "application/json",
                "Set-Cookie": `goatbot_session=${token}; Path=/; Max-Age=${7 * 24 * 60 * 60}; HttpOnly; Secure; SameSite=Lax`
            }
        });

    } catch (err) {
        console.error("Verify error:", err);
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
