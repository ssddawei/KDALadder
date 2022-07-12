// include storage.js
let SysLocalStorage = window.localStorage;

class LocalStorage extends Storage {
  constructor() {
    super();
    this.load();
  }
  load() {
    this.ladder = JSON.parse(SysLocalStorage.getItem("ladder")) || [];
    this.matches = JSON.parse(SysLocalStorage.getItem("matches")) || [];
  }
  save() {
    SysLocalStorage.setItem("ladder", JSON.stringify(this.ladder));
    SysLocalStorage.setItem("matches", JSON.stringify(this.matches));
  }
}