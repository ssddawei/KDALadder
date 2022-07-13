// include storage.js
let SysLocalStorage = window.localStorage;

class LocalStorage extends Storage {
  constructor(index) {
    super(index);
    this.load();
  }
  load() {
    this.ladder = JSON.parse(SysLocalStorage.getItem(`ladder-${this.index}`)) || new Ladder();
    this.matches = JSON.parse(SysLocalStorage.getItem(`matches-${this.index}`)) || [];
  }
  save() {
    SysLocalStorage.setItem(`ladder-${this.index}`, JSON.stringify(this.ladder));
    SysLocalStorage.setItem(`matches-${this.index}`, JSON.stringify(this.matches));
  }
  delete() {
    SysLocalStorage.removeItem(`ladder-${this.index}`);
    SysLocalStorage.removeItem(`matches-${this.index}`);
  }
}