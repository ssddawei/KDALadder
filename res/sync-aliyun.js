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
    let upstreamName = `data-${now.getFullYear()}-${now.getMonth()+1}.json`;
    let upstream = JSON.parse(await store.get(upstreamName).catch(e=>{
      if(e && e.code == 'NoSuchKey')
        return null;
      else
        throw e;
    }));
    
    // merge
    Object.assign(upstream.matches, storage.matches);
    Object.assign(upstream.ladder, storage.ladder);

    // save to upstream
    await store.put(upstreamName, new OSS.Buffer([JSON.stringify(upstream)]));

    // apply to local
    this.storage.matches = upstream.matches;
    this.storage.ladder = upstream.ladder;

    // save to local
    this.storage.save();
  }
}