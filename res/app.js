
class MatchController {
  match = new Match();
  get ready() {
    return !!(this.match.personGroup.filter(i=>i).length == 4)
  }
  get started() {
    return this.aScore > 0 || this.bScore > 0;
  }
  get aGroup() {
    return this.match.personGroup.slice(0,2);
  }
  get bGroup() {
    return this.match.personGroup.slice(2,4);
  }
  get aScore() {
    return this.match.scores.filter(i => this.aGroup.indexOf(i.kill) >= 0).length +
      this.match.scores.filter(i => this.bGroup.indexOf(i.death) >= 0).length
  }
  get bScore() {
    return this.match.scores.filter(i => this.bGroup.indexOf(i.kill) >= 0).length +
      this.match.scores.filter(i => this.aGroup.indexOf(i.death) >= 0).length
  }
  constructor(aGroupOrMatch = [], bGroup = []) {
    if(aGroupOrMatch instanceof Match || aGroupOrMatch.scores) {
      this.match = aGroupOrMatch;
    } else {
      this.match.personGroup = [...aGroupOrMatch, ...bGroup];
    }
  }
  goal(person) {
    let assistGroup = this.aGroup.indexOf(person) >= 0? this.aGroup: this.bGroup;
    let assist = assistGroup.filter(i => i != person)[0];
    this.match.scores.push(new GameScore(person, null, assist))
  }
  loss(person) {
    this.match.scores.push(new GameScore(null, person))
  }
  revert() {
    this.match.scores.length && this.match.scores.length --;
  }
  kda(person) {
    if(+person < 4) {
      person = this.match.personGroup[+person];
    }
    let kill = this.match.scores.filter(i => i.kill == person).length;
    let death = this.match.scores.filter(i => i.death == person).length;
    let assist = this.match.scores.filter(i => i.assist == person).length;
    let win = this.match.scores.filter(i => i.win && i.win.indexOf(person) >= 0).length;
    let loss = this.match.scores.filter(i => i.loss && i.loss.indexOf(person) >= 0).length;
    let kda = {
      kill, death, assist, win, loss
    }
    kda.score = ALG.PersonScore(kda);
    return kda;
  }
  mvp() {
    if(!this.started)return;
    let kda = this.match.personGroup.map(this.kda.bind(this));
    if(this.match.scores.filter(i => i.win || i.loss).length <= 0) {
      if(this.aScore > this.bScore) {
        kda[0].win = kda[1].win = 1;
      } else {
        kda[2].win = kda[3].win = 1;
      }
    }
    return this.match.personGroup[Object.entries(kda).sort((a,b)=>b[1].score-a[1].score)[0][0]];
  }
  loser() {
    if(!this.started)return;
    let kda = this.match.personGroup.map(this.kda.bind(this));
    if(this.match.scores.filter(i => i.win || i.loss).length <= 0) {
      if(this.aScore > this.bScore) {
        kda[0].win = kda[1].win = 1;
      } else {
        kda[2].win = kda[3].win = 1;
      }
    }
    return this.match.personGroup[Object.entries(kda).sort((a,b)=>a[1].score-b[1].score)[0][0]];
  }
  static LadderEvolve(ladder, person, kda) {
    let item = ladder.filter(i => i.person == person)[0];
    if(!item) {
      ladder.push(item = { 
        person
      });
    }
    item.kill = (+item.kill || 0) + (+kda.kill || 0);
    item.death = (+item.death || 0) + (+kda.death || 0);
    item.assist = (+item.assist || 0) + (+kda.assist || 0);
    item.win = (+item.win || 0) + (+kda.win || 0);
    item.loss = (+item.loss || 0) + (+kda.loss || 0);
    item.score = ALG.PersonScore(item);
    ladder.sort((a,b) => b.score - a.score)
  }
  save() {
    let storage = new LocalStorage("current");
    storage.data = {"current": this.match}
    storage.save();
  }
  load() {
    let storage = new LocalStorage("current");
    if(storage.data.current)
      this.match = storage.data.current;
  }
  ladder() {
    let ladder = new Ladder();
    ladder.beginTime = this.match.beginTime;
    ladder.endTime = this.match.endTime2;
    ladder.matchCount = 1;
    ladder.matchTotalTimeSec = Math.floor((this.match.endTime2 - this.match.beginTime)/1000);
    
    // update to local ladder
    MatchController.LadderEvolve(ladder.ladder, this.aGroup[0], this.kda(this.aGroup[0]));
    MatchController.LadderEvolve(ladder.ladder, this.aGroup[1], this.kda(this.aGroup[1]));
    MatchController.LadderEvolve(ladder.ladder, this.bGroup[0], this.kda(this.bGroup[0]));
    MatchController.LadderEvolve(ladder.ladder, this.bGroup[1], this.kda(this.bGroup[1]));
    
    return ladder;
  }
  async end() {

    if(!this.ready || !this.started) return;

    let storage = new LocalStorage();
    let sync = new AliyunSyncData(storage, new LocalStorage("remote"));

    // check sync key
    if(!sync.key) {
      let key = await $prompt("同步到云端，请输入密钥");
      if(!key) {
        if(!await $confirm("不保存到云端，确认？")){
          return;
        }
      } else {
        if(!sync.saveKey(key)){
          await $alert("密钥格式错误");
          return;
        }
      }
    }

    let now = new Date();

    // add win/loss score
    if( this.aScore > this.bScore ) {
      this.match.scores.push(new MatchScore(this.aGroup, this.bGroup))
    } else {
      this.match.scores.push(new MatchScore(this.bGroup, this.aGroup))
    }
    
    // save to local
    (storage.data[$dateString(now)] = storage.data[$dateString(now)] || [])
      .push(this.match);

    (storage.ladder[$seasonString(now)] = storage.ladder[$seasonString(now)] || [])
      .push(this.ladder());

    storage.save();

    // sync
    await sync.sync();

    new LocalStorage("current").delete();

    this.match = new Match();

    return $dateString(now);
  }
}

class LadderController {
  syncData = new AliyunSyncData(null, new LocalStorage("remote"));

  get storage() {
    return this.syncData.local
  }
  get remote() {
    return this.syncData.remoteCache;
  }
  async seasonLadder(season = $seasonString(new Date())) {
    let ladder = this.remote.ladder[season];
    if(!ladder) {
      let dateOfSession = season
        .replace("season1", "1-1")
        .replace("season2", "4-1")
        .replace("season3", "7-1")
        .replace("season4", "10-1")
      await this.syncData.loadRemote(new Date(dateOfSession));
      ladder = this.remote.ladder[season];
    }
    
    // sum each date' ladders of this season
    return ladder.reduce((pre, nxt) => {
      nxt.ladder.forEach(kda => {
        MatchController.LadderEvolve(pre.ladder, kda.person, kda);
      })
      return pre;
    }, {ladder:[]})
  }
  async dateLadder(date = $dateString(new Date())) {
    let season = $seasonString(new Date(date));
    let ladder = this.remote.ladder[season];
    
    if(!ladder) {
      await this.syncData.loadRemote(new Date(date));
      ladder = this.remote.ladder[season];
    }
    
    // not found in REMOTE , and found match , calc ladder
    if(!ladder.length && this.dateMatch(date)) {
      let matches = await this.dateMatch(date);
      ladder =  matches.map(match => {
        return new MatchController(match).ladder()
      })

      // cache
      this.syncData.remoteCache.ladder[season] = ladder;
      this.syncData.remoteCache.save();
    }

    // sum each date' ladders of this season
    return ladder.reduce((pre, nxt) => {
      if($dateString(new Date(nxt.beginTime)) == date)
        nxt.ladder.forEach(kda => {
          MatchController.LadderEvolve(pre.ladder, kda.person, kda);
        })
      return pre;
    }, {ladder:[]})
  }
  async dateMatch(date = $dateString(new Date())) {
    let match = this.remote.data[date];
    if(!match) {
      await this.syncData.loadRemote(new Date(date));
    }
    return this.remote.data[date];
  }

  async sync() {
    await this.syncData.sync();
  }
}

/*
  Bind ConnectWebrtc to UI
*/
class ConnectController {
  conn;
  status;
  mode;
  onData;
  constructor(mode, onData) {
    this.mode = mode; // act as "server" or "client"
    this.onData = onData; // on data receive

    $sel(".connect").addEventListener("click", async () => {

      // check sync key
      if(!new AliyunSyncData().key) {
        let key = await $prompt("多端连接，请输入密钥");
        if(!key) {
          return;
        } else {
          if(!new AliyunSyncData().saveKey(key)){
            await $alert("密钥格式错误");
            return;
          }
        }
      }

      this.refreshUI("loading");
      this.connect();
    });
  }
  refreshUI(status) {
    this.status = status;

    $sel(".connect").classList.remove("loading");
    $sel(".connect").classList.remove("error");
    $sel(".connect").classList.remove("done");
    if(typeof(this.status) == "string")
      $sel(".connect").classList.add(this.status);
    else if(typeof(this.status) == "object")
      $sel(".connect").classList.add(...this.status);
  }
  async connect() {
    if(this.conn){ 
      this.conn.close();
      this.conn = undefined;
    }

    // create ConnectWebrtc, set receiveCallback, errorCallback
    let conn = this.conn = new ConnectWebrtc(new AliyunSyncData(), 
      (msg) => {
        this.refreshUI("done");
        if(msg.indexOf("hi") == 0)return;
        
        let data = JSON.parse(msg);
        this.onData && this.onData(data);
      }, (err) => {
        this.refreshUI(["error", "done"]);
      });

    try{

      // invoke offer/answer to connect Client/Server
      if(this.mode == "server")
        await conn.offer();
      else if(this.mode == "client")
        await conn.answer();
      else
        throw new Error("wrong mode: " + mode)

      conn.send("hi " + this.mode);

    } catch(e) {
      if(conn != this.conn) {
        // conn released
      }
      else if(e && e.message == "timeout") {
        this.refreshUI();
      } else if(e && e.message == "canceled") {
        // conn canceled
      } else {
        $alert("连接失败: " + e)
        this.refreshUI("error");
      }
    }


  }
  send(action, data) {
    this.status == "done" && this.conn.send(JSON.stringify({action, data}));
  }
}
class Menu {
  constructor() {
    $sel(".menuBtn").addEventListener("click", ()=>{
      $sel(".menu").classList.add("show");
    })
    $sel(".menu").addEventListener("click", ()=>{
      $sel(".menu").classList.remove("show");
    })
    let btn = $sel(".menuBtn");
    let menu = $sel(".menu");
    document.addEventListener("click", function(e){
      if(e.path.filter(i=>i==btn||i==menu).length == 0){
        $sel(".menu").classList.remove("show");
      }
    })
  }
}

class ListChooser {
  chooseCallback;
  loader;
  constructor(loader) {
    this.loader = loader;
    $sel("div.dataList .list").addEventListener("click", (e) => {
      let data = e.path.filter(i => i.dataset && i.dataset["data"])[0];
      this.select(data && data.dataset["data"]);
    })
    $sel("div.dataList .cancelBtn").addEventListener("click", () => {
      this.cancel();
    });
  }
  async refreshUI() {
    let datas = this.loader && await this.loader() || [];
    let tpl = $sel("#DataListItem").innerHTML;
    $sel("div.dataList > .list").innerHTML = datas.map(d => {
      let data = d.data || d;
      let subtitle = d.subtitle || "";
      return tpl.replace(/{{data}}/g, data)
        .replace("{{subtitle}}", subtitle);
    }).join("");
  }
  select(date) {
    this.chooseCallback && this.chooseCallback(date);
    $sel(".dataList").classList.remove("show")
  }
  cancel() {
    this.chooseCallback && this.chooseCallback();
    $sel(".dataList").classList.remove("show")
  }
  choose() {
    this.refreshUI();
    $sel(".dataList").classList.add("show")

    return new Promise(done => {
      this.chooseCallback = done;
    })
  };
}
