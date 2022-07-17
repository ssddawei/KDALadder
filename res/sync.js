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
  remote = {ladder:{}, data:{}};
  local = {ladder:{}, data:{}};
  constructor(storage) {
    this.local = storage;
  }
  sync() {}
}