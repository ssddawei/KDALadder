export let CONFIG = {
  ServerUrl: `http://${location.hostname}:8080`,
  AliyunOSSKey: {
    region: "oss-cn-guangzhou",
    bucket: "kdaladder-dev",
    timeout: "5s",
    // accessKeyId: "",
    // accessKeySecret: "",
  },
  DataUrl: "//kdaladder-dev.oss-cn-guangzhou.aliyuncs.com/"
}