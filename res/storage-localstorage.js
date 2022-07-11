// include storage.js
let SysLocalStoratge = window.LocalStorage;

class LocalStorage extends Storage {
  load() {
    this.ladder = JSON.parse(SysLocalStoratge.getItem("ladder"));
    this.matches = JSON.parse(SysLocalStoratge.getItem("matches"));
  }
  save() {
    SysLocalStoratge.setItem("ladder", JSON.stringify(this.ladder));
    SysLocalStoratge.setItem("matches", JSON.stringify(this.matches));
  }
}