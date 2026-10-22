FROM node:20-alpine

WORKDIR /app
COPY automation/package.json ./package.json
COPY automation/server ./server
COPY automation/public ./public

ENV PORT=8080 AUTOMATION_DATA_DIR=/data
RUN mkdir -p /data && chown -R node:node /app /data
USER node
EXPOSE 8080
VOLUME ["/data"]

CMD ["node", "server/index.js"]
