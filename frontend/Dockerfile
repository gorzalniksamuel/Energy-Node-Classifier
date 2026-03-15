FROM node:22-alpine AS build

# Create app directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy  frontend code
COPY . .

# Build the production bundle (Vite: outputs to /app/dist)
RUN npm run build

# Run stage (static file server)
FROM nginx:1.27-alpine

# Copy built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 in the container
EXPOSE 80

# Run nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
