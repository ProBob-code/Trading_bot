#!/bin/bash

# GodBotTrade Local Development Setup
# ==================================

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up GodBotTrade Local Environment...${NC}"

# 1. Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠️ Docker not found. Please install Docker to use the containerized database.${NC}"
    exit 1
fi

# 2. Start MySQL container
echo -e "${GREEN}📦 Starting MySQL database container...${NC}"
docker compose up -d mysql

# 3. Wait for MySQL to be ready
echo -e "${YELLOW}⏳ Waiting for MySQL to be ready (health check)...${NC}"
until [ "`docker inspect -f {{.State.Health.Status}} trading-bot-db`"=="healthy" ]; do
    sleep 2
done

echo -e "${GREEN}✅ MySQL is ready and healthy!${NC}"

# 4. Success message
echo -e "\n${BLUE}=================================================${NC}"
echo -e "${GREEN}🎉 Local environment is ready!${NC}"
echo -e "You can now run the server:"
echo -e "   ${YELLOW}./start_server.sh${NC}"
echo -e "\nOr start the full stack in Docker:"
echo -e "   ${YELLOW}docker compose up --build -d${NC}"
echo -e "${BLUE}=================================================${NC}"
