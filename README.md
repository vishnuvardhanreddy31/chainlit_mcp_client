# Chainlit MCP App

AI Assistant with Model Context Protocol (MCP) integration using Google Gemini.

## Features

- Google Gemini AI integration
- MCP (Model Context Protocol) support
- Tool calling capabilities
- Context-aware conversations

## Deployment

This app is configured for deployment on Render.

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key

### Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env` file
3. Run: `chainlit run app.py`

## MCP Integration

The app supports MCP connections and can use tools from various MCP servers like Stripe, Linear, and others.
