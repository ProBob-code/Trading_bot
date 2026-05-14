export async function onRequestPost(context) {
    const { request, env } = context;
    const secret = env.DB_PROXY_SECRET || "god-bot-trade-secret-2026";
    
    const authHeader = request.headers.get("Authorization");
    if (authHeader !== `Bearer ${secret}`) {
        return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401, headers: { "Content-Type": "application/json" } });
    }
    
    try {
        const { query, params, method = "all" } = await request.json();
        
        if (!query) {
            return new Response(JSON.stringify({ error: "Missing query" }), { status: 400, headers: { "Content-Type": "application/json" } });
        }
        
        const db = env.trading_bot_v2;
        if (!db) {
            return new Response(JSON.stringify({ error: "Database binding missing" }), { status: 500, headers: { "Content-Type": "application/json" } });
        }
        
        const stmt = db.prepare(query);
        let preparedStmt = stmt;
        if (params && params.length > 0) {
            preparedStmt = stmt.bind(...params);
        }
        
        let result;
        if (method === "run") {
            result = await preparedStmt.run();
        } else {
            result = await preparedStmt.all();
        }
        
        return new Response(JSON.stringify(result), { headers: { "Content-Type": "application/json" } });
    } catch (e) {
        return new Response(JSON.stringify({ error: e.message }), { status: 500, headers: { "Content-Type": "application/json" } });
    }
}
