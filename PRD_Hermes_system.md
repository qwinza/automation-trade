Product Requirements Document (PRD)

Project Name: Hermes Multi-Agent Trading Orchestrator
Target Environment: Local Development -> Production (VPS Biznet Gio Cloud)

1. System Overview

Hermes is an Automated Directional Trading orchestrator. It receives technical market data via TradingView webhooks (acting as an MCP - Model Context Protocol), synthesizes the data using a Multi-Agent AI System (The Bull, The Bear, The Judge) based on Smart Money Concepts (SMC), and logs the final trading decisions into a MySQL database. AI requests are routed through a local 9Router proxy/reseller API.

2. Tech Stack & Architecture

Backend Framework: Python 3.10+, FastAPI, Uvicorn, Pydantic

AI Provider / Proxy: Anthropic SDK routing to 9Router / mwapi.dev (Custom Base URL)

Database ORM: Prisma ORM (Python Client)

Relational Database: MySQL (User: hermes, handling auth_socket bypass)

State Management: Redis (localhost:6379) for caching and concurrency locks.

3. Development Phases

Phase 1: Environment & Database Setup (Prisma + MySQL)

Goal: Initialize the project structure and setup the database schema.

Initialize a Python virtual environment and install dependencies: fastapi, uvicorn, pydantic, redis, python-dotenv, anthropic, prisma.

Initialize Prisma (prisma init).

Define the schema.prisma:

Model TradeDecision:

id (String, UUID, @id)

asset (String)

timeframe (String)

action (String - BUY/SELL/HOLD)

confidence (Int)

reasoning (Text)

market_context (Json)

created_at (DateTime, @default(now()))

Run Prisma migrations and generate the client.

Phase 2: Data Ingestion (TradingView Webhook)

Goal: Create the FastAPI entry point to receive market signals.

In src/main.py, initialize FastAPI and Redis client.

Define a Pydantic model TradingViewPayload:

asset (str)

timeframe (str)

price (float)

structure (str)

liquidity (str)

order_block (str)

fvg (str)

Create POST /tv-webhook endpoint to accept this payload and format it into a comprehensive market_data string.

Phase 3: Multi-Agent Orchestration (Hermes Brain)

Goal: Implement the LLM logic using Anthropic SDK.

Create src/agents/prompts.py containing:

BULL_AGENT_PROMPT: Focuses on Long setups (SSL sweep, Discount OB, Bullish FVG).

BEAR_AGENT_PROMPT: Focuses on Short setups (BSL sweep, Premium OB, Bearish FVG).

JUDGE_AGENT_PROMPT: Chief Risk Manager. Demands RR >= 1:2. Outputs ONLY valid JSON: {"action": "BUY"|"SELL"|"HOLD", "confidence": int, "reasoning": "str"}.

Create src/agents/orchestrator.py:

Configure Anthropic client with custom .env variables (CLAUDE_API_KEY and ANTHROPIC_BASE_URL).

Function run_multi_agent_analysis(asset, market_data):

Calls Bull Agent (Temperature 0.7).

Calls Bear Agent (Temperature 0.7).

Calls Judge Agent (Temperature 0.1) passing both arguments.

Parses and returns the JSON.

Phase 4: State Management & Database Logging

Goal: Prevent duplicate trades and save decisions.

Integrate Redis in src/main.py:

Before triggering the orchestrator, check Redis key trade_lock:{asset}:{timeframe}.

If exists, ignore webhook (prevents signal spam).

If not, set lock with a 15-minute expiration (TTL).

Integrate Prisma inside POST /tv-webhook:

After the Judge returns a decision, asynchronously save the record to the TradeDecision MySQL table.

Phase 5: Deployment & Integration Strategy

Goal: Prepare for VPS execution.

Ensure all hardcoded credentials are moved to .env.

Create .env.example defining expected keys (DATABASE_URL, CLAUDE_API_KEY, ANTHROPIC_BASE_URL, MODEL_NAME).

Note for Agent: The connection to 9Router in production will use http://localhost:20128 or the specific reseller URL provided by the user.

4. Execution Directives for AI Agent

Step-by-Step: Do not attempt to code all 5 phases in one massive output. Ask the user to confirm completion of Phase 1 before moving to Phase 2.

Code Quality: Ensure all FastAPI routes use async/await appropriately, especially when interacting with Prisma ORM and Anthropic's async client.

Error Handling: Implement try-except blocks around the LLM calls and Database queries to prevent server crashes.