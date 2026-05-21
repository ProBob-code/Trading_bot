/**
 * POST /api/auth/register
 * Register with mobile number, generate mock OTP.
 */
export async function onRequestPost(context) {
    const { env, request } = context;

    try {
        const db = env.trading_bot_v2;
        if (!db) {
            return jsonResponse({ success: false, error: "Database not available" }, 500);
        }

        const { mobile } = await request.json();

        if (!mobile) {
            return jsonResponse({ success: false, error: "Mobile number is required" }, 400);
        }

        // Check if user already exists with this mobile
        let user = await db.prepare(
            "SELECT id FROM users WHERE mobile = ?"
        ).bind(mobile).first();

        if (!user) {
            // Create new user row
            await db.prepare(
                "INSERT INTO users (mobile) VALUES (?)"
            ).bind(mobile).run();

            user = await db.prepare(
                "SELECT id FROM users WHERE mobile = ?"
            ).bind(mobile).first();
        }

        if (!user) {
            return jsonResponse({ success: false, error: "Database error — could not create user" }, 500);
        }

        // Generate 6-digit mock OTP
        const otp = String(Math.floor(100000 + Math.random() * 900000));

        // Store OTP
        await db.prepare(
            "UPDATE users SET otp = ? WHERE id = ?"
        ).bind(otp, user.id).run();

        // In production, you'd send this via SMS. For demo, log it.
        console.log(`[MOCK OTP] For ${mobile}: ${otp}`);

        return jsonResponse({
            success: true,
            message: "OTP sent to mobile",
            otp_sent: true,
            // Include OTP in response for demo/testing (remove in production)
            _demo_otp: otp
        });

    } catch (err) {
        console.error("Register error:", err);
        return jsonResponse({ success: false, error: "Server error" }, 500);
    }
}

function jsonResponse(data, status = 200) {
    return new Response(JSON.stringify(data), {
        status,
        headers: { "Content-Type": "application/json" }
    });
}
