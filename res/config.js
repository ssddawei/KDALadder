export let CONFIG = {
  ServerUrl: `http://${location.host}`,
  AliyunOSSKey: {
    region: "oss-cn-guangzhou",
    bucket: "kdaladder-dev",
    timeout: "5s",
    // accessKeyId: "",
    // accessKeySecret: "",
  },
  DataUrl: "//kdaladder-dev.oss-cn-guangzhou.aliyuncs.com/"
}