// include sync.js
// include aliyunsdk
class AliyunSyncData extends SyncData {
  get key(){
    let key = window.localStorage.getItem("key");
    if(key){
      return JSON.parse(key);
    }
  }
  saveKey(keyString) {
    let keys = keyString.split("/");
    if(keys.length != 2)return false;

    this.key = Object.assign({}, CONFIG.AliyunOSSKey, {
      accessKeyId: keys[0],
      accessKeySecret: keys[1],
    });

    window.localStorage.setItem("key", this.key);
    return true;
  }
  async sync(idx = ALG.StorageIndex()) {

    if(!this.key) {
      console.warn("no key exist");
    }

    const store = this.key && new OSS(this.key);

    //
    // sync matches
    //
    let storage = this.storages[idx];
    if(!storage) {
      console.error("no storage to sync")
      return;
    }

    let upstreamName = `data-${idx}.json`;
    let upstreamUrl = CONFIG.DataUrl.data + upstreamName;
    let upstream = await $fetch(upstreamUrl);
    if(!upstream) {
      upstream = {
        matches: [],
        ladder: new Ladder()
      }
    }
    
    // merge
    Object.assign(upstream.matches, storage.matches);
    Object.assign(upstream.ladder, storage.ladder);

    // save to upstream
    store && await store.put(upstreamName, new OSS.Buffer(JSON.stringify(upstream)));

    // apply to local
    storage.matches = upstream.matches;
    storage.ladder = upstream.ladder;

    // save to local
    storage.save();

    //
    // sync total-ladder
    //
    upstreamUrl = CONFIG.DataUrl.ladder;
    upstream = await $fetch(upstreamUrl);

    if(!upstream) {
      upstream = new Ladder()
    }

    // update ladder
    storage.ladder.ladder.forEach(l => {
      MatchController.LadderEvolve(upstream.ladder, l.person, l);
    })

    upstream.beginTime = upstream.beginTime || storage.ladder.beginTime;
    upstream.endTime = storage.ladder.endTime;
    upstream.matchCount = (+upstream.matchCount||0) + (+storage.ladder.matchCount||0);
    upstream.matchTotalTimeSec = (+upstream.matchTotalTimeSec||0) + (+storage.ladder.matchTotalTimeSec||0);

    // save to upstream
    store && await store.put(upstreamName, new OSS.Buffer(JSON.stringify(upstream)));

    if(this.totalLadder) {
      // apply to local
      this.totalLadder.ladder = upstream;
      // save to local
      this.totalLadder.save();
    }

    return store? "remote": "local"
  }
}