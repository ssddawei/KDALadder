// include storage.js
let SysLocalStorage = window.localStorage;

class LocalStorage extends Storage {
  constructor(prefix) {
    super(prefix);
    this.load();
  }
  load() {
    this.ladder = JSON.parse(SysLocalStorage.getItem(`${this.prefix}-ladder`)) || {};
    this.data = JSON.parse(SysLocalStorage.getItem(`${this.prefix}-data`)) || {};
  }
  save() {
    SysLocalStorage.setItem(`${this.prefix}-ladder`, JSON.stringify(this.ladder));
    SysLocalStorage.setItem(`${this.prefix}-data`, JSON.stringify(this.data));
  }
  delete() {
    SysLocalStorage.removeItem(`${this.prefix}-ladder`);
    SysLocalStorage.removeItem(`${this.prefix}-data`);
  }
}