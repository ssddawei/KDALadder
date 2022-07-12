// include sync.js
// include aliyunsdk
class AliyunSyncData extends SyncData {
  
  async sync() {
    const store = new OSS({
      region: 'oss-cn-guangzhou',
      accessKeyId: 'LTAI5tAxkvizwCkYfVBnzx92',
      accessKeySecret: 'QkpBi19GzIv3Vp6QPrrHsRHjU1DgpR',
      bucket: 'kdaladder',
    });

    let now = new Date();
    // let upstreamName = `data-${now.getFullYear()}-${now.getMonth()+1}.json`;
    let upstreamName = `data.json`;
    let upstreamResult = await store.get(upstreamName).catch(e=>{
      if(e && e.code == 'NoSuchKey')
        return null;
      else
        throw e;
    })


    let upstream = JSON.parse(upstreamResult.content.toString());

    if(!upstream) {
      upstream = {
        matches: [],
        ladder: []
      }
    }
    
    // merge
    Object.assign(upstream.matches, this.storage.matches);
    Object.assign(upstream.ladder, this.storage.ladder);

    // console.log(new OSS.Buffer(JSON.stringify(upstream)))

    // save to upstream
    await store.put(upstreamName, new OSS.Buffer(JSON.stringify(upstream)));

    // apply to local
    this.storage.matches = upstream.matches;
    this.storage.ladder = upstream.ladder;

    // save to local
    this.storage.save();
  }
}