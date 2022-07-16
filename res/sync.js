class SyncData {
  // storage for each date
  // example:
  /*
    {
      ladder: {
        "2022-7-12":[Ladder],
        "2022-7-13":[Ladder],
      }
      data: {
        "2022-7-12":[Match],
        "2022-7-13":[Match],
      }
    }

  */
  remote = {ladder:{}, data:{}};
  local = {ladder:{}, data:{}};
  constructor() {
  }
  sync(idx) {}
}