/*

*/
const STUNS = [
  // "stun:stun.xten.com:3478",
  // "stun:stun.voipbuster.com:3478",
  // "stun:stun.sipgate.net:3478",
  // "stun:stun.ekiga.net:3478",
  // "stun:stun.ideasip.com:3478",
  // "stun:stun.schlund.de:3478",
  // "stun:stun.voiparound.com:3478",
  // "stun:stun.voipbuster.com:3478",
  // "stun:stun.voipstun:stunt.com:3478",
  // "stun:stun.counterpath.com:3478",
  // "stun:stun.1und1.de:3478",
  // "stun:stun.gmx.net:3478",
  // "stun:stun.callwithus.com:3478",
  // "stun:stun.counterpath.net:3478",
  // "stun:stun.internetcalls.com:3478",
  // "stun:numb.viagenie.ca:3478",
]

class ConnectWebrtc {
  sync; // use to exchange SDP ( such aliyun oss AliyunSyncData )
  receiveCallback; // webrtc RTCDataChannel message receiver
  errorCallback; // webrtc RTCDataChannel error 
  pc; // RTCPeerConnection
  channel; // RTCDataChannel
  waitCandidateMax = 3;
  constructor(sync, receiveCallback, errorCallback) {
    this.sync = sync;
    this.receiveCallback = receiveCallback;
    this.errorCallback = errorCallback;
  }
  close() {
    if(this.pc) {
      this.pc.close();
      this.pc = undefined;
    }
  }
  // Server invoke offer, wait client to answer
  async offer() {

    await this.sync.save("offer.sdp");
    await this.sync.save("answer.sdp");

    let pc = this.pc = new RTCPeerConnection({
      "iceCandidatePoolSize":1,
      "iceServers":STUNS.map(i => ({urls:i}))

    }, null);

    // create channel
    this.channel = pc.createDataChannel("default");
    let open = new Promise(o => {
      this.channel.onopen = (event) => {
        console.log("onopen")
        o()
      }
    })
    this.channel.onmessage = (event) => {
      // console.log(event.data);
      this.receiveCallback && this.receiveCallback(event.data);
    }
    this.channel.onerror = (e) => {
      this.pc && this.errorCallback && this.errorCallback(e);
    }
    this.channel.onclose = (e) => {
      this.pc && this.errorCallback && this.errorCallback(e);
    }

    // wait candidate
    let candidate = new Promise(o => {
      let count = 0;
      pc.onicecandidate = (e)=>{
        console.log("oncandidate", e.candidate)
        if(!e.candidate) o();
        else if(++count >= this.waitCandidateMax) o();
      };
    })

    // createOffer

    let offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    await candidate;
    
    offer = pc.localDescription;
    await this.sync.save("offer.sdp", offer);

    // wait answer
    let answer;
    let timeout = 30000;
    const SPAN = 2000;
    do{
      if(timeout <= 0) throw new Error("timeout");
      if(!this.pc) throw new Error("canceled");
      await new Promise(o => setTimeout(o, SPAN)); // sleep 1 sec
      answer = await this.sync.load("answer.sdp");
      timeout -= SPAN;
    } while(!answer);

    await pc.setRemoteDescription(answer);

    // wait open
    await open;

    await this.sync.save("offer.sdp");
    await this.sync.save("answer.sdp");
  }
  // Client invoke answer, wait server to response
  async answer() {
    let pc = this.pc = new RTCPeerConnection({
      "iceCandidatePoolSize":1,
      "iceServers":STUNS.map(i => ({urls:i}))
    }, null);

    // create channel
    let open = new Promise(o => {
      pc.ondatachannel = (event) => {
        this.channel = event.channel;
        this.channel.onopen = (event) => {
          console.log("onopen")
          o();
        }
        this.channel.onmessage = (event) => {
          // console.log(event.data);
          this.receiveCallback && this.receiveCallback(event.data);
        }
        this.channel.onerror = (e) => {
          this.pc && this.errorCallback && this.errorCallback(e);
        }
        this.channel.onclose = (e) => {
          this.pc && this.errorCallback && this.errorCallback(e);
        }
      }
    })

    // wait candidate
    let candidate = new Promise(o => {
      let count = 0;
      pc.onicecandidate = (e)=>{
        console.log("oncandidate", e.candidate)
        if(!e.candidate) o();
        else if(++count >= this.waitCandidateMax) o();
      };
    })

    // wait offer
    let offer;
    let timeout = 30000;
    const SPAN = 2000;
    do{
      if(timeout <= 0) throw new Error("timeout");
      if(!this.pc) throw new Error("canceled");
      await new Promise(o => setTimeout(o, SPAN)); // sleep 1 sec
      offer = await this.sync.load("offer.sdp");
      timeout -= SPAN;
    } while(!offer);

    await pc.setRemoteDescription(offer);

    // create answer
    let answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    await candidate;

    await this.sync.save("answer.sdp", pc.localDescription);

    // wait open
    await open;
  }
  // send datachannel message
  async send(msg) {
    this.channel.send(msg);
  }
}