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

    let key = Object.assign({}, CONFIG.AliyunOSSKey, {
      accessKeyId: keys[0],
      accessKeySecret: keys[1],
    });

    window.localStorage.setItem("key", JSON.stringify(key));
    return true;
  }
  get lastMatchIndex() {
    return this.matchIndex.filter(i=>i!=ALG.StorageIndex()).slice(-1)[0];
  }
  get lastStorage() {
    let currentStorage = this.storages[ALG.StorageIndex()]
    if(currentStorage.matches.length) {
      return currentStorage;
    } else  {
      return this.storages[this.lastMatchIndex] || currentStorage;
    }
  }
  async sync(idx = ALG.StorageIndex()) {

    if(!this.key) {
      console.warn("no key exist");
    }

    const store = this.key && new OSS(this.key);

    this.matchIndex = await $fetch(CONFIG.DataUrl.matches) || [];

    //
    // sync matches
    //
    let storage = this.storages[idx];
    if(!storage) {
      storage = this.storages[idx] = new LocalStorage(idx);
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
    if(store && this.matchIndex.slice(-1)[0] != idx) {
      this.matchIndex.push(idx);
      this.matchIndex = Array.from(new Set(this.matchIndex)).sort((a,b)=>new Date(a)-new Date(b));
      await store.put("matches.json", new OSS.Buffer(JSON.stringify(this.matchIndex)));
    }

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
    store && await store.put("ladder.json", new OSS.Buffer(JSON.stringify(upstream)));

    if(this.totalLadder) {
      // apply to local
      this.totalLadder.ladder = upstream;
      // save to local
      this.totalLadder.save();
    }

    return store? "remote": "local"
  }
}