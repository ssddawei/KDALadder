// include sync.js
// include aliyunsdk
// ali-sdk: https://github.com/ali-sdk/ali-oss
class AliyunSyncData extends SyncData {
  static LadderURL(season, relative) {
    return (relative? "": CONFIG.DataUrl) + `ladder-${season}.json`;
  }
  static DataURL(date, relative) {
    return (relative? "": CONFIG.DataUrl) + `data-${date}.json`;
  }

  get key(){
    let key = window.localStorage.getItem("key");
    if(key){
      return JSON.parse(key);
    }
  }
  saveKey(keyString) {
    let keys = keyString.split("/");
    if(keys.length != 2)return false;

    let key = Object.assign({}, CONFIG.AliyunOSSKey, {
      accessKeyId: keys[0],
      accessKeySecret: keys[1],
    });

    window.localStorage.setItem("key", JSON.stringify(key));
    return true;
  }
  async loadRemote(date) {
    let seasonStr = $seasonString(date);
    let dateStr = $dateString(date);
    this.remote.ladder[seasonStr] = await $fetch(AliyunSyncData.LadderURL(seasonStr)) || [];
    this.remote.data[dateStr] = await $fetch(AliyunSyncData.DataURL(dateStr)) || [];

    // cache
    Object.assign(this.remoteCache.ladder, this.remote.ladder);
    Object.assign(this.remoteCache.data, this.remote.data);
    this.remoteCache.save();
  }
  async saveRemote() {
    if(!this.key) {
      console.warn("no key exist");
      return;
    }

    const store = new OSS(this.key);

    await Promise.all([

      ...Object.entries(this.local.ladder).map(async ([season,ladder]) => {
        let rLadder = this.remote.ladder[season];
        let unsyncLadder = ladder.filter(i => {
          return !rLadder || !rLadder.find(r => r.beginTime == i.beginTime)
        });
        if(unsyncLadder.length) { 
          let data = JSON.stringify(unsyncLadder).slice(1, -1);
          if(rLadder && rLadder.length)
            data = "," + data;

          await store.append(AliyunSyncData.LadderURL(season, true), new OSS.Buffer(data), {
            position: rLadder? JSON.stringify(rLadder).length - 2: 0
          });

          rLadder.splice(rLadder.length, 0, ...unsyncLadder);
        }
      }),
      ...Object.entries(this.local.data).map(async ([date,match]) => {
        let rMatch = this.remote.data[date];
        let unsyncData = match.filter(i => {
          return !rMatch || !rMatch.find(r => r.beginTime == i.beginTime)
        });
        if(unsyncData.length) { 
          let data = JSON.stringify(unsyncData).slice(1, -1);
          if(rMatch && rMatch.length)
            data = "," + data;

          await store.append(AliyunSyncData.DataURL(date, true), new OSS.Buffer(data), {
            position: rMatch? JSON.stringify(rMatch).length - 2: 0
          });

          rMatch.splice(rMatch.length, 0, ...unsyncData);
        }
      }),

    ]);

    this.remoteCache.save();

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