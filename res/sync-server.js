// include sync.js
// include aliyunsdk
// ali-sdk: https://github.com/ali-sdk/ali-oss

let DEFAULT_SERVER_URL = CONFIG.ServerUrl;
class ServerAPI {
  
  constructor(ServerURL) {
    this.ServerURL = ServerURL || DEFAULT_SERVER_URL
  }
  call(url, params) {
    params = params || {}
    if(!params.groupCode)
      params.groupCode = this.groupCode;
    return $fetch(url, {
      method: "POST",
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    })
  }
  postMatch(matchData, ladderData) {
    return this.call(new URL("/v1/group/match", this.ServerURL).toString(), {
      matchData,
      ladderData
    })
  }
  login(groupCode) {
    return this.call(new URL("/v1/group/hash", this.ServerURL).toString(), {
      groupCode
    })
  }
  register(groupCode, groupName, inviteCode) {
    return this.call(new URL("/v1/group/new", this.ServerURL).toString(), {
      groupCode, groupName, inviteCode
    })
  }
  postInfo(groupName, groupCode) {
    return this.call(new URL("/v1/group/info", this.ServerURL).toString(), {
      newGroupCode: groupCode, groupName
    })
  }
}
class ServerSyncData extends SyncData {

  constructor(storage = null, remoteCacheStorage = new LocalStorage("remote")) {
    super(storage, remoteCacheStorage)
    this.api = new ServerAPI()
    this.autologin()
  }
  LadderURL(season) {
    return new URL(`${this.groupCodeHashPath}/ladder-${season}.json`,DEFAULT_SERVER_URL).toString();
  }
  DataURL(date) {
    return new URL(`${this.groupCodeHashPath}/data-${date}.json`,DEFAULT_SERVER_URL).toString();
  }
  autologin() {
    if(localStorage.getItem("gc")) {
      let {groupCode, groupCodeHashPath} = JSON.parse(atob(localStorage.getItem("gc")));
      this.api.groupCode = groupCode;
      this.groupCodeHashPath = groupCodeHashPath;
    }
  }
  async register(groupCode, groupName, inviteCode) {
    let groupCodeHashPath = (await this.api.register(groupCode, groupName, inviteCode)).groupCodeHashPath;
    this.api.groupCode = groupCode;
    this.groupCodeHashPath = groupCodeHashPath;
    localStorage.setItem("gc", btoa(JSON.stringify(
      {
        groupCode, groupCodeHashPath
      }
    )));
  }
  async login(groupCode) {
    let groupCodeHashPath = (await this.api.login(groupCode)).groupCodeHashPath;
    this.api.groupCode = groupCode;
    this.groupCodeHashPath = groupCodeHashPath;
    localStorage.setItem("gc", btoa(JSON.stringify(
      {
        groupCode, groupCodeHashPath
      }
    )));
  }
  async loadRemote(date) {
    let seasonStr = $seasonString(date);
    let dateStr = $dateString(date);
    this.remote.ladder[seasonStr] = await $fetch(this.LadderURL(seasonStr), {toArray:true}) || [];
    this.remote.data[dateStr] = await $fetch(this.DataURL(dateStr), {toArray:true}) || [];

    // cache
    Object.assign(this.remoteCache.ladder, this.remote.ladder);
    Object.assign(this.remoteCache.data, this.remote.data);
    this.remoteCache.save();
  }
  async saveRemote() {

    if(!this.local){
      return;
    }
    

    await Promise.all([
      ...Object.entries(this.local.ladder).map(async ([season,ladder]) => {
        let rLadder = this.remote.ladder[season];
        let unsyncData = ladder.filter(i => {
          return !rLadder || !rLadder.find(r => r.beginTime == i.beginTime)
        });
        
        // save to server
        await this.api.postMatch(null, unsyncData)

        rLadder.splice(rLadder.length, 0, ...unsyncData);
        this.remoteCache.save();

      }),
      ...Object.entries(this.local.data).map(async ([date,match]) => {
        let rMatch = this.remote.data[date];
        let unsyncData = match.filter(i => {
          return !rMatch || !rMatch.find(r => r.beginTime == i.beginTime)
        });

        // save to server
        await this.api.postMatch(unsyncData, null)

        rMatch.splice(rMatch.length, 0, ...unsyncData);
        this.remoteCache.save();
      })
    ])

    this.local.delete();

    return true;
  }
  async sync() {

    let now = new Date();

    await this.loadRemote(now);

    let savedToRemove = await this.saveRemote();

    return savedToRemove? "remote": "local"
  }
}