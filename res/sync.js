class SyncData {
  // storage for each date
  // example:
  /*
    {
      "2022-7-12":Storage,
      "2022-7-13":Storage,
    }

  */
  storages = {};
  totalLadder = new LocalStorage("total");
  matchIndex = [];
  constructor() {
  }
  sync(idx) {}
}