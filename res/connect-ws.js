import { CONFIG } from './config.js';
export class ConnectWebsocket {
  constructor(groupCode, receiveCallback, errorCallback) {
    this.ws = new WebSocket(
      CONFIG.ServerUrl
        .replace(/^https:/i, "wss:")
        .replace(/^http:/i, "ws:")
    )
    this.ws.addEventListener("open", () => {
      // login
      this.ws.send(groupCode)
    });

    this.ws.addEventListener("error", errorCallback);
    this.ws.addEventListener("close", (msg) => errorCallback(msg));

    let loginDone = false
    this.ws.addEventListener("message", (msg) => {

      // first msg is login result
      if(!loginDone && msg.data) {
        receiveCallback && receiveCallback("hi")
        loginDone = true;
        return;
      }

      receiveCallback && receiveCallback(msg.data)

    })
  }
  close() {
    this.ws && this.ws.close(1000, "user close");
  }
  send(msg) {
    if(typeof(msg) == "string") {
      this.ws.send(msg);
    } else {
      this.ws.send(JSON.stringify(msg));
    }
  }
}