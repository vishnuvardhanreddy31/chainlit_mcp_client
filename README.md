# 🤖 CodeCrafter AI - Intelligent Coding Assistant

An advanced AI-powered coding assistant built with **Google Gemini** and **Model Context Protocol (MCP)**, providing intelligent code assistance, tool integration, and context-aware conversations.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

## ✨ Features

### 🎯 Core Capabilities
- **AI-Powered Assistance**: Leverages Google Gemini 2.0 Flash for intelligent code generation and problem-solving
- **MCP Integration**: Seamlessly connects with MCP servers for extended tool functionality
- **Context-Aware**: Maintains conversation history for coherent multi-turn interactions
- **Real-Time Streaming**: Provides immediate feedback with streaming responses
- **Tool Calling**: Automatically invokes appropriate tools when needed

### 🎨 User Experience
- **Modern UI**: Clean, responsive interface with custom styling
- **Mobile-First Design**: Fully responsive layout that works on all devices
- **Dark/Light Themes**: Automatic theme switching with user preferences
- **Visual Feedback**: Loading states, animations, and clear status indicators
- **Error Handling**: User-friendly error messages with helpful context

### 🔧 Technical Features
- **Type Safety**: Comprehensive type hints for better code quality
- **Error Recovery**: Robust error handling with graceful degradation
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Performance Optimized**: Efficient tool conversion and API calls
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Google Gemini API key
- (Optional) MCP server configurations

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishnuvardhanreddy31/chainlit_mcp_client.git
   cd chainlit_mcp_client
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   chainlit run app.py
   ```

5. **Access the UI**
   
   Open your browser to `http://localhost:8000`

## 📖 Usage Guide

### Basic Conversation
Simply type your coding questions or requests in the chat interface:

```
User: How do I create a REST API in Python?
CodeCrafter: I'll help you create a REST API in Python using FastAPI...
```

### Tool Usage
When MCP tools are available, CodeCrafter will automatically use them:

```
User: Check my recent expenses
CodeCrafter: [Calls expense_tracker tool] Here are your recent expenses...
```

### Best Practices
1. **Be Specific**: Provide clear context and requirements
2. **Iterate**: Refine your requests based on responses
3. **Use Tools**: Leverage MCP integrations for enhanced capabilities
4. **Review Code**: Always review generated code before using in production

## 🔌 MCP Integration

### Connecting MCP Servers

Configure MCP servers in your Chainlit configuration:

```toml
[mcp.servers]
  [mcp.servers.your_server]
  command = "path/to/mcp/server"
  args = ["--config", "config.json"]
```

### Supported Tool Types
- Development tools (linters, formatters, testers)
- API integrations (Stripe, Linear, GitHub)
- Database connections
- Custom business logic tools

### Creating Custom Tools
Refer to the [MCP Documentation](https://modelcontextprotocol.io) for creating custom tools.

## ⚙️ Configuration

### Chainlit Settings

Edit `.chainlit/config.toml` to customize:

```toml
[UI]
name = "CodeCrafter AI"
default_theme = "light"
layout = "wide"
custom_css = "/public/style.css"

[features]
edit_message = true
```

### Gemini Settings

Modify `app.py` to adjust Gemini parameters:

```python
GENERATION_CONFIG = {
    "temperature": 0.7,      # Creativity (0.0-1.0)
    "max_output_tokens": 2048,  # Response length
    "top_p": 0.95,          # Diversity
}
```

## 🎨 Customization

### Custom CSS
The application uses custom CSS for enhanced styling. Modify `public/style.css` to customize:

- Color schemes and themes
- Typography and spacing
- Animations and transitions
- Responsive breakpoints

### System Prompt
Customize the AI's behavior by editing `SYSTEM_PROMPT` in `app.py`:

```python
SYSTEM_PROMPT = """Your custom instructions here..."""
```

## 🌐 Deployment

### Deploy to Render

1. **Fork the repository**
2. **Connect to Render**
3. **Set environment variables**:
   - `GEMINI_API_KEY`: Your API key
   - `PYTHON_VERSION`: 3.11
4. **Deploy**: Automatic deployment from main branch

### Deploy to Other Platforms

The application is compatible with any platform supporting Python web apps:
- Heroku: Use included `Procfile`
- Vercel: Configure build commands
- AWS/GCP: Use container deployment
- Docker: Create a Dockerfile (example below)

#### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing

### Manual Testing
1. Start the application
2. Test basic conversations
3. Verify tool calling (if MCP servers connected)
4. Check error handling
5. Test responsive design on different devices

### Automated Testing
(Add tests as the project grows)

```bash
pytest tests/
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: "GEMINI_API_KEY not found"
- **Solution**: Ensure `.env` file exists with valid API key

**Issue**: "Tool not found in any MCP connection"
- **Solution**: Verify MCP server is running and connected

**Issue**: "Model blocked response"
- **Solution**: Rephrase query to avoid potential safety triggers

### Debug Mode
Enable detailed logging:

```bash
export CHAINLIT_DEBUG=true
chainlit run app.py
```

### Getting Help
- Check the [Chainlit Documentation](https://docs.chainlit.io)
- Review [Gemini API Docs](https://ai.google.dev/docs)
- Open an issue on GitHub

## 📊 Architecture

```
┌─────────────────┐
│   User (Web UI) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Chainlit App   │
│   (app.py)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    v         v
┌───────┐  ┌──────────┐
│Gemini │  │MCP Servers│
│  API  │  │  (Tools)  │
└───────┘  └──────────┘
```

### Component Overview

1. **Frontend**: Chainlit web UI with custom styling
2. **Backend**: Python application handling logic
3. **AI Model**: Google Gemini for generation
4. **Tools**: MCP servers for extended functionality

## 🔒 Security

### Best Practices
- Never commit API keys or secrets
- Use environment variables for sensitive data
- Validate all user inputs
- Keep dependencies updated
- Review generated code before execution

### Environment Variables
Store sensitive data in `.env` (not tracked by git):
```env
GEMINI_API_KEY=your_secret_key
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Chainlit](https://chainlit.io) - For the amazing chat UI framework
- [Google Gemini](https://ai.google.dev) - For powerful AI capabilities
- [MCP](https://modelcontextprotocol.io) - For tool integration protocol

## 📧 Contact

- GitHub: [@vishnuvardhanreddy31](https://github.com/vishnuvardhanreddy31)
- Issues: [GitHub Issues](https://github.com/vishnuvardhanreddy31/chainlit_mcp_client/issues)

---

**Built with ❤️ by the community**
