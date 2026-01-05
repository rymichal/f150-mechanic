# Future Features

## Epic 1: Context Tracking -- DONE
Track and display token usage to help users understand context consumption and manage conversation length.

**Implementation Steps:**
- Install and integrate tiktoken library
- Create token counting utility function
- Calculate tokens for each message (user + assistant)
- Track cumulative context usage throughout conversation
- Display context percentage/usage on each message
- Add visual indicator (e.g., progress bar or percentage) in UI
- Consider adding warnings when approaching context limits

## Epic 2: Conversation Memory. -- DONE
Implement persistent conversation storage to enable multi-session continuity and conversation history.

**Implementation Steps:**
- Design conversation data schema (messages, metadata, timestamps)
- Choose storage mechanism (file-based, database, etc.)
- Implement conversation save/load functionality
- Add conversation listing and selection UI
- Create conversation metadata (title, created/updated dates, summary)
- Implement conversation search/filter capabilities
- Add conversation export/import features

## Epic 3: MCP (Model Context Protocol) Connection
Integrate MCP to enable standardized communication with external tools and data sources.

**Implementation Steps:**
- Research MCP specification and requirements
- Install MCP client libraries
- Configure MCP server connections
- Implement MCP message handling
- Add tool/resource discovery via MCP
- Create MCP-compatible tool adapters
- Add error handling and reconnection logic
- Test with common MCP servers (filesystem, database, etc.)

## Epic 4: Database Connection
Enable direct database connectivity for data retrieval and manipulation.

**Implementation Steps:**
- Choose database adapter/ORM (e.g., Prisma, TypeORM, SQLAlchemy)
- Design connection configuration schema
- Implement connection pooling and management
- Create query execution interface
- Add query result formatting
- Implement SQL injection protection
- Add support for multiple database types (PostgreSQL, MySQL, SQLite)
- Create database exploration tools (schema inspection, table listing)
- Add query result caching (optional)

## Epic 5: Serverless Function Triggers
Enable triggering of Azure Functions or AWS Lambda for external task execution.

**Implementation Steps:**
- Set up authentication for Azure/AWS services
- Install Azure Functions SDK and/or AWS Lambda SDK
- Create function invocation wrapper utilities
- Implement request/response handling
- Add async function execution support
- Create function result polling/webhook handling
- Add timeout and retry logic
- Implement logging for function invocations
- Create configuration for function endpoints
- Add support for passing parameters to functions
- Handle different function response formats

## Epic 6: Trigger Email