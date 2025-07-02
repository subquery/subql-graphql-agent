# SubQuery GraphQL Agent Frontend

A modern React TypeScript frontend for the Multi-Project SubQuery GraphQL Agent server.

## Features

- 🚀 **Project Registration**: Register SubQuery projects via IPFS CID
- 💬 **Interactive Chat**: Chat with project-specific GraphQL agents
- ⚙️ **Configuration Management**: Edit project prompts and capabilities
- 🔄 **Real-time Streaming**: Support for streaming chat responses
- 📱 **Responsive Design**: Modern UI that works on all devices
- 🎯 **Project Switching**: Easily switch between multiple projects

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- SubQuery GraphQL Agent backend running on port 8000

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Usage

### 1. Register a Project

1. Click "Register New Project" in the sidebar
2. Enter the IPFS CID of your SubQuery project manifest
3. The system will automatically fetch the manifest and schema
4. Your project will appear in the project list

### 2. Configure Project Settings

1. Select a project from the list
2. Click the "Configuration" tab
3. Edit domain name, capabilities, and decline message
4. Save your changes

### 3. Chat with the Agent

1. Select a project and go to the "Chat" tab
2. Ask questions about the indexed data
3. Use streaming mode for real-time responses
4. Clear chat history as needed

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ProjectRegistration.tsx
│   │   ├── ProjectList.tsx
│   │   ├── ProjectConfig.tsx
│   │   └── ChatInterface.tsx
│   ├── hooks/              # Custom React hooks
│   │   ├── useProjects.ts
│   │   └── useChat.ts
│   ├── services/           # API services
│   │   └── api.ts
│   ├── types/              # TypeScript types
│   │   └── index.ts
│   ├── lib/                # Utilities
│   │   └── utils.ts
│   ├── styles/             # CSS styles
│   │   └── globals.css
│   ├── App.tsx             # Main app component
│   └── main.tsx            # Entry point
├── public/                 # Static assets
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## API Integration

The frontend communicates with the backend through:

- **REST API**: Project management, configuration
- **Server-Sent Events**: Streaming chat responses
- **React Query**: Data fetching, caching, mutations

### API Endpoints Used

- `POST /register` - Register new project
- `GET /projects` - List projects  
- `GET /projects/{cid}` - Get project config
- `PATCH /projects/{cid}` - Update project config
- `POST /{cid}/chat/completions` - Chat with project
- `GET /health` - Server health check

## Customization

### Styling

The project uses Tailwind CSS with a custom design system. Modify `src/styles/globals.css` and `tailwind.config.js` to customize the appearance.

### API Configuration

Update `VITE_API_URL` in `.env` to point to your backend server.

### Components

All components are modular and can be easily customized:

- `ProjectRegistration` - CID input and validation
- `ProjectList` - Project selection interface  
- `ProjectConfig` - Settings editor
- `ChatInterface` - Chat UI with streaming

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript compiler

### Code Quality

- **TypeScript** for type safety
- **ESLint** for code linting
- **Prettier** for code formatting (via Tailwind)
- **React Query** for server state management

## Deployment

### Static Hosting

Build the project and deploy the `dist` folder to any static hosting service:

```bash
npm run build
# Deploy contents of dist/ folder
```

### Docker

Create a `Dockerfile`:

```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment Variables

For production, set:

```bash
VITE_API_URL=https://your-backend-api.com
```

## Browser Support

- Chrome/Edge 88+
- Firefox 78+  
- Safari 14+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

Same license as the parent project.