# Use an official Node.js runtime as a parent image
FROM node:20-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json to the container
COPY package.json ./

# Install production dependencies
RUN npm install --omit=dev

# The MCP server listens on the PORT environment variable, which Cloud Run sets automatically.
# We expose 8080, which is the default port Cloud Run uses.
EXPOSE 8080

# Define the command to run your app
CMD [ "npm", "start" ]