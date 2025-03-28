
# Dockerfile for Web Interface (Frontend)
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx config (if you have one)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 

    CMD wget --quiet --tries=1 --spider http://localhost:80/health || exit 1

# Start nginx server
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
