// include sync.js
// include aliyunsdk
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
  }
  async saveRemote() {
    if(!this.key) {
      console.warn("no key exist");
      return;
    }

    const store = new OSS(this.key);

    await Promise.all(

      Object.entries(this.local.ladder).map(async ([season,ladder]) => {
        let unsyncLadder = ladder.filter(i => {
          return !this.remote[season] || !this.remote[season].ladder.find(r => r.beginTime == i.beginTime)
        });

        let data = JSON.stringify(unsyncLadder).slice(1, -1);
        if(this.remote[season] && this.remote[season].ladder.length)
          data = "," + data;

        console.log("store.append", unsyncLadder);
        // await store.append(AliyunSyncData.LadderURL(season, true), new OSS.Buffer(data));
        
      })

    );

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