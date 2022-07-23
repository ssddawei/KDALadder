class SyncData {
  // storage for each date
  // example:
  /*
    {
      ladder: {
        "2022-season1":[Ladder],
        "2022-season2":[Ladder],
      }
      data: {
        "2022-7-12":[Match],
        "2022-7-13":[Match],
      }
    }

  */
  // remote = {ladder:{}, data:{}};
  // local = {ladder:{}, data:{}};
  // remoteCache;
  constructor(storage, remoteCacheStorage) {
    this.remote = {ladder:{}, data:{}};
    this.local = storage;
    this.remoteCache = remoteCacheStorage;
  }
  sync() {}
}