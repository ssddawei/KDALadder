// include sync.js
// include aliyunsdk
// ali-sdk: https://github.com/ali-sdk/ali-oss

let DEFAULT_SERVER_URL = ""
class ServerAPI {
  
  constructor(ServerURL) {
    this.ServerURL = ServerURL || DEFAULT_SERVER_URL
  }
  async call(url, params) {
    params.groupCode = "test2"
    return await $fetch(url, {
      method: "POST",
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    })
  }
  async postMatch(matchData, ladderData) {
    return await this.call(this.ServerURL + "v1/group/match", {
      matchData,
      ladderData
    })
  }
}
class ServerSyncData extends SyncData {

    constructor(storage, remoteCacheStorage) {
      super(storage, remoteCacheStorage)
      this.api = new ServerAPI("http://localhost/")
    }
    static LadderURL(season, relative) {
      return (relative? "": CONFIG.DataUrl) + `ladder-${season}.json`;
    }
    static DataURL(date, relative) {
      return (relative? "": CONFIG.DataUrl) + `data-${date}.json`;
    }
    static OtherURL(name, relative) {
      return (relative? "": CONFIG.DataUrl) + name;
    }
  
    async loadRemote(date) {
      let seasonStr = $seasonString(date);
      let dateStr = $dateString(date);
      this.remote.ladder[seasonStr] = await $fetch(ServerSyncData.LadderURL(seasonStr)) || [];
      this.remote.data[dateStr] = await $fetch(ServerSyncData.DataURL(dateStr)) || [];
  
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
  
      if(!this.key) {
        console.warn("no key exist");
      }
  
      let now = new Date();
  
      await this.loadRemote(now);
  
      let savedToRemove = await this.saveRemote();
  
      return savedToRemove? "remote": "local"
    }
  }