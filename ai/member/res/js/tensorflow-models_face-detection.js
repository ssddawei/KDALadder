/**
    * @license
    * Copyright 2023 Google LLC. All Rights Reserved.
    * Licensed under the Apache License, Version 2.0 (the "License");
    * you may not use this file except in compliance with the License.
    * You may obtain a copy of the License at
    *
    * http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing, software
    * distributed under the License is distributed on an "AS IS" BASIS,
    * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    * See the License for the specific language governing permissions and
    * limitations under the License.
    * =============================================================================
    */
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?t(exports,require("@mediapipe/face_detection"),require("@tensorflow/tfjs-core"),require("@tensorflow/tfjs-converter")):"function"==typeof define&&define.amd?define(["exports","@mediapipe/face_detection","@tensorflow/tfjs-core","@tensorflow/tfjs-converter"],t):t((e=e||self).faceDetection={},e.globalThis,e.tf,e.tf)}(this,(function(e,t,i,n){"use strict";var o=function(){return o=Object.assign||function(e){for(var t,i=1,n=arguments.length;i<n;i++)for(var o in t=arguments[i])Object.prototype.hasOwnProperty.call(t,o)&&(e[o]=t[o]);return e},o.apply(this,arguments)};function r(e,t,i,n){return new(i||(i=Promise))((function(o,r){function a(e){try{u(n.next(e))}catch(e){r(e)}}function s(e){try{u(n.throw(e))}catch(e){r(e)}}function u(e){var t;e.done?o(e.value):(t=e.value,t instanceof i?t:new i((function(e){e(t)}))).then(a,s)}u((n=n.apply(e,t||[])).next())}))}function a(e,t){var i,n,o,r,a={label:0,sent:function(){if(1&o[0])throw o[1];return o[1]},trys:[],ops:[]};return r={next:s(0),throw:s(1),return:s(2)},"function"==typeof Symbol&&(r[Symbol.iterator]=function(){return this}),r;function s(r){return function(s){return function(r){if(i)throw new TypeError("Generator is already executing.");for(;a;)try{if(i=1,n&&(o=2&r[0]?n.return:r[0]?n.throw||((o=n.return)&&o.call(n),0):n.next)&&!(o=o.call(n,r[1])).done)return o;switch(n=0,o&&(r=[2&r[0],o.value]),r[0]){case 0:case 1:o=r;break;case 4:return a.label++,{value:r[1],done:!1};case 5:a.label++,n=r[1],r=[0];continue;case 7:r=a.ops.pop(),a.trys.pop();continue;default:if(!(o=a.trys,(o=o.length>0&&o[o.length-1])||6!==r[0]&&2!==r[0])){a=0;continue}if(3===r[0]&&(!o||r[1]>o[0]&&r[1]<o[3])){a.label=r[1];break}if(6===r[0]&&a.label<o[1]){a.label=o[1],o=r;break}if(o&&a.label<o[2]){a.label=o[2],a.ops.push(r);break}o[2]&&a.ops.pop(),a.trys.pop();continue}r=t.call(e,a)}catch(e){r=[6,e],n=0}finally{i=o=0}if(5&r[0])throw r[1];return{value:r[0]?r[1]:void 0,done:!0}}([r,s])}}}var s=["rightEye","leftEye","noseTip","mouthCenter","rightEarTragion","leftEarTragion"];var u={modelType:"short",runtime:"mediapipe",maxFaces:1};var h=function(){function e(e){var i=this;this.width=0,this.height=0,this.selfieMode=!1,this.faceDetectorSolution=new t.FaceDetection({locateFile:function(t,i){if(e.solutionPath){var n=e.solutionPath.replace(/\/+$/,"");return"".concat(n,"/").concat(t)}return"".concat(i,"/").concat(t)}}),this.faceDetectorSolution.setOptions({selfieMode:this.selfieMode,model:e.modelType}),this.faceDetectorSolution.onResults((function(e){if(i.height=e.image.height,i.width=e.image.width,i.faces=[],null!==e.detections)for(var t=0,n=e.detections;t<n.length;t++){var o=n[t];i.faces.push(i.normalizedToAbsolute(o.landmarks,(r=o.boundingBox,a=void 0,s=void 0,u=void 0,a=r.xCenter-r.width/2,s=a+r.width,u=r.yCenter-r.height/2,{xMin:a,xMax:s,yMin:u,yMax:u+r.height,width:r.width,height:r.height})))}var r,a,s,u}))}return e.prototype.normalizedToAbsolute=function(e,t){var i=this;return{keypoints:e.map((function(e,t){return{x:e.x*i.width,y:e.y*i.height,name:s[t]}})),box:{xMin:t.xMin*this.width,yMin:t.yMin*this.height,xMax:t.xMax*this.width,yMax:t.yMax*this.height,width:t.width*this.width,height:t.height*this.height}}},e.prototype.estimateFaces=function(e,t){return r(this,void 0,void 0,(function(){var n,o;return a(this,(function(r){switch(r.label){case 0:return t&&t.flipHorizontal&&t.flipHorizontal!==this.selfieMode&&(this.selfieMode=t.flipHorizontal,this.faceDetectorSolution.setOptions({selfieMode:this.selfieMode})),e instanceof i.Tensor?(o=ImageData.bind,[4,i.browser.toPixels(e)]):[3,2];case 1:return n=new(o.apply(ImageData,[void 0,r.sent(),e.shape[1],e.shape[0]])),[3,3];case 2:n=e,r.label=3;case 3:return e=n,[4,this.faceDetectorSolution.send({image:e})];case 4:return r.sent(),[2,this.faces]}}))}))},e.prototype.dispose=function(){this.faceDetectorSolution.close()},e.prototype.reset=function(){this.faceDetectorSolution.reset(),this.width=0,this.height=0,this.faces=null,this.selfieMode=!1},e.prototype.initialize=function(){return this.faceDetectorSolution.initialize()},e}();function c(e){return r(this,void 0,void 0,(function(){var t,i;return a(this,(function(n){switch(n.label){case 0:return t=function(e){if(null==e)return o({},u);var t=o({},e);return t.runtime="mediapipe",null==t.modelType&&(t.modelType=u.modelType),null==t.maxFaces&&(t.maxFaces=u.maxFaces),t}(e),[4,(i=new h(t)).initialize()];case 1:return n.sent(),[2,i]}}))}))}function l(e,t,i,n){var o=e.width,r=e.height,a=n?-1:1,s=Math.cos(e.rotation),u=Math.sin(e.rotation),h=e.xCenter,c=e.yCenter,l=1/t,d=1/i,f=new Array(16);return f[0]=o*s*a*l,f[1]=-r*u*l,f[2]=0,f[3]=(-.5*o*s*a+.5*r*u+h)*l,f[4]=o*u*a*d,f[5]=r*s*d,f[6]=0,f[7]=(-.5*r*s-.5*o*u*a+c)*d,f[8]=0,f[9]=0,f[10]=o*l,f[11]=0,f[12]=0,f[13]=0,f[14]=0,f[15]=1,function(e){if(16!==e.length)throw new Error("Array length must be 16 but got ".concat(e.length));return[[e[0],e[1],e[2],e[3]],[e[4],e[5],e[6],e[7]],[e[8],e[9],e[10],e[11]],[e[12],e[13],e[14],e[15]]]}(f)}function d(e){return e instanceof i.Tensor?{height:e.shape[0],width:e.shape[1]}:{height:e.height,width:e.width}}function f(e){return e instanceof i.Tensor?e:i.browser.fromPixels(e)}function p(e,t){i.util.assert(0!==e.width,(function(){return"".concat(t," width cannot be 0.")})),i.util.assert(0!==e.height,(function(){return"".concat(t," height cannot be 0.")}))}function m(e,t){var n=function(e,t,i,n){var o=t-e,r=n-i;if(0===o)throw new Error("Original min and max are both ".concat(e,", range cannot be 0."));var a=r/o;return{scale:a,offset:i-e*a}}(0,255,t[0],t[1]);return i.tidy((function(){return i.add(i.mul(e,n.scale),n.offset)}))}function x(e,t,n){var o=t.outputTensorSize,r=t.keepAspectRatio,a=t.borderMode,s=t.outputTensorFloatRange,u=d(e),h=function(e,t){return t?{xCenter:t.xCenter*e.width,yCenter:t.yCenter*e.height,width:t.width*e.width,height:t.height*e.height,rotation:t.rotation}:{xCenter:.5*e.width,yCenter:.5*e.height,width:e.width,height:e.height,rotation:0}}(u,n),c=function(e,t,i){if(void 0===i&&(i=!1),!i)return{top:0,left:0,right:0,bottom:0};var n=t.height,o=t.width;p(t,"targetSize"),p(e,"roi");var r,a,s=n/o,u=e.height/e.width,h=0,c=0;return s>u?(r=e.width,a=e.width*s,c=(1-u/s)/2):(r=e.height/s,a=e.height,h=(1-s/u)/2),e.width=r,e.height=a,{top:c,left:h,right:h,bottom:c}}(h,o,r),x=l(h,u.width,u.height,!1),g=i.tidy((function(){var t=f(e),n=i.tensor2d(function(e,t,i){return p(i,"inputResolution"),[1/i.width*e[0][0]*t.width,1/i.height*e[0][1]*t.width,e[0][3]*t.width,1/i.width*e[1][0]*t.height,1/i.height*e[1][1]*t.height,e[1][3]*t.height,0,0]}(x,u,o),[1,8]),r="zero"===a?"constant":"nearest",h=i.image.transform(i.expandDims(i.cast(t,"float32")),n,"bilinear",r,0,[o.height,o.width]);return null!=s?m(h,s):h}));return{imageTensor:g,padding:c,transformationMatrix:x}}function g(e){null==e.reduceBoxesInLowestLayer&&(e.reduceBoxesInLowestLayer=!1),null==e.interpolatedScaleAspectRatio&&(e.interpolatedScaleAspectRatio=1),null==e.fixedAnchorSize&&(e.fixedAnchorSize=!1);for(var t=[],i=0;i<e.numLayers;){for(var n=[],o=[],r=[],a=[],s=i;s<e.strides.length&&e.strides[s]===e.strides[i];){var u=y(e.minScale,e.maxScale,s,e.strides.length);if(0===s&&e.reduceBoxesInLowestLayer)r.push(1),r.push(2),r.push(.5),a.push(.1),a.push(u),a.push(u);else{for(var h=0;h<e.aspectRatios.length;++h)r.push(e.aspectRatios[h]),a.push(u);if(e.interpolatedScaleAspectRatio>0){var c=s===e.strides.length-1?1:y(e.minScale,e.maxScale,s+1,e.strides.length);a.push(Math.sqrt(u*c)),r.push(e.interpolatedScaleAspectRatio)}}s++}for(var l=0;l<r.length;++l){var d=Math.sqrt(r[l]);n.push(a[l]/d),o.push(a[l]*d)}var f=0,p=0;if(e.featureMapHeight.length>0)f=e.featureMapHeight[i],p=e.featureMapWidth[i];else{var m=e.strides[i];f=Math.ceil(e.inputSizeHeight/m),p=Math.ceil(e.inputSizeWidth/m)}for(var x=0;x<f;++x)for(var g=0;g<p;++g)for(var v=0;v<n.length;++v){var w={xCenter:(g+e.anchorOffsetX)/p,yCenter:(x+e.anchorOffsetY)/f,width:0,height:0};e.fixedAnchorSize?(w.width=1,w.height=1):(w.width=o[v],w.height=n[v]),t.push(w)}i=s}return t}function y(e,t,i,n){return 1===n?.5*(e+t):e+(t-e)*i/(n-1)}function v(e,t){var i=t[0],n=t[1];return[i*e[0]+n*e[1]+e[3],i*e[4]+n*e[5]+e[7]]}function w(e){return i.tidy((function(){var t=function(e){return i.tidy((function(){return[i.slice(e,[0,0,0],[1,-1,1]),i.slice(e,[0,0,1],[1,-1,-1])]}))}(e),n=t[0],o=t[1];return{boxes:i.squeeze(o),logits:i.squeeze(n)}}))}function M(e,t,n,o){return r(this,void 0,void 0,(function(){var o,r,s,u,h;return a(this,(function(a){switch(a.label){case 0:return e.sort((function(e,t){return Math.max.apply(Math,t.score)-Math.max.apply(Math,e.score)})),o=i.tensor2d(e.map((function(e){return[e.locationData.relativeBoundingBox.yMin,e.locationData.relativeBoundingBox.xMin,e.locationData.relativeBoundingBox.yMax,e.locationData.relativeBoundingBox.xMax]}))),r=i.tensor1d(e.map((function(e){return e.score[0]}))),[4,i.image.nonMaxSuppressionAsync(o,r,t,n)];case 1:return[4,(s=a.sent()).array()];case 2:return u=a.sent(),h=e.filter((function(e,t){return u.indexOf(t)>-1})),i.dispose([o,r,s]),[2,h]}}))}))}function b(e,t,n){return r(this,void 0,void 0,(function(){var o,r,s,u,h;return a(this,(function(a){switch(a.label){case 0:return o=e[0],r=e[1],s=function(e,t,n){return i.tidy((function(){var o,r,a,s;n.reverseOutputOrder?(r=i.squeeze(i.slice(e,[0,n.boxCoordOffset+0],[-1,1])),o=i.squeeze(i.slice(e,[0,n.boxCoordOffset+1],[-1,1])),s=i.squeeze(i.slice(e,[0,n.boxCoordOffset+2],[-1,1])),a=i.squeeze(i.slice(e,[0,n.boxCoordOffset+3],[-1,1]))):(o=i.squeeze(i.slice(e,[0,n.boxCoordOffset+0],[-1,1])),r=i.squeeze(i.slice(e,[0,n.boxCoordOffset+1],[-1,1])),a=i.squeeze(i.slice(e,[0,n.boxCoordOffset+2],[-1,1])),s=i.squeeze(i.slice(e,[0,n.boxCoordOffset+3],[-1,1]))),r=i.add(i.mul(i.div(r,n.xScale),t.w),t.x),o=i.add(i.mul(i.div(o,n.yScale),t.h),t.y),n.applyExponentialOnBoxSize?(a=i.mul(i.exp(i.div(a,n.hScale)),t.h),s=i.mul(i.exp(i.div(s,n.wScale)),t.w)):(a=i.mul(i.div(a,n.hScale),t.h),s=i.mul(i.div(s,n.wScale),t.h));var u=i.sub(o,i.div(a,2)),h=i.sub(r,i.div(s,2)),c=i.add(o,i.div(a,2)),l=i.add(r,i.div(s,2)),d=i.concat([i.reshape(u,[n.numBoxes,1]),i.reshape(h,[n.numBoxes,1]),i.reshape(c,[n.numBoxes,1]),i.reshape(l,[n.numBoxes,1])],1);if(n.numKeypoints)for(var f=0;f<n.numKeypoints;++f){var p=n.keypointCoordOffset+f*n.numValuesPerKeypoint,m=void 0,x=void 0;n.reverseOutputOrder?(m=i.squeeze(i.slice(e,[0,p],[-1,1])),x=i.squeeze(i.slice(e,[0,p+1],[-1,1]))):(x=i.squeeze(i.slice(e,[0,p],[-1,1])),m=i.squeeze(i.slice(e,[0,p+1],[-1,1])));var g=i.add(i.mul(i.div(m,n.xScale),t.w),t.x),y=i.add(i.mul(i.div(x,n.yScale),t.h),t.y);d=i.concat([d,i.reshape(g,[n.numBoxes,1]),i.reshape(y,[n.numBoxes,1])],1)}return d}))}(r,t,n),u=i.tidy((function(){var e=o;return n.sigmoidScore?(null!=n.scoreClippingThresh&&(e=i.clipByValue(o,-n.scoreClippingThresh,n.scoreClippingThresh)),e=i.sigmoid(e)):e})),[4,S(s,u,n)];case 1:return h=a.sent(),i.dispose([s,u]),[2,h]}}))}))}function S(e,t,i){return r(this,void 0,void 0,(function(){var n,o,r,s,u,h,c,l,d,f,p,m;return a(this,(function(a){switch(a.label){case 0:return n=[],[4,e.data()];case 1:return o=a.sent(),[4,t.data()];case 2:for(r=a.sent(),s=0;s<i.numBoxes;++s)if(!(null!=i.minScoreThresh&&r[s]<i.minScoreThresh||(u=s*i.numCoords,h=T(o[u+0],o[u+1],o[u+2],o[u+3],r[s],i.flipVertically,s),(c=h.locationData.relativeBoundingBox).width<0||c.height<0))){if(i.numKeypoints>0)for((l=h.locationData).relativeKeypoints=[],d=i.numKeypoints*i.numValuesPerKeypoint,f=0;f<d;f+=i.numValuesPerKeypoint)p=u+i.keypointCoordOffset+f,m={x:o[p+0],y:i.flipVertically?1-o[p+1]:o[p+1]},l.relativeKeypoints.push(m);n.push(h)}return[2,n]}}))}))}function T(e,t,i,n,o,r,a){return{score:[o],ind:a,locationData:{relativeBoundingBox:{xMin:t,yMin:r?1-i:e,xMax:n,yMax:r?1-e:i,width:n-t,height:i-e}}}}var C="https://tfhub.dev/mediapipe/tfjs-model/face_detection/short/1",z={reduceBoxesInLowestLayer:!1,interpolatedScaleAspectRatio:1,featureMapHeight:[],featureMapWidth:[],numLayers:4,minScale:.1484375,maxScale:.75,inputSizeHeight:128,inputSizeWidth:128,anchorOffsetX:.5,anchorOffsetY:.5,strides:[8,16,16,16],aspectRatios:[1],fixedAnchorSize:!0},O={reduceBoxesInLowestLayer:!1,interpolatedScaleAspectRatio:0,featureMapHeight:[],featureMapWidth:[],numLayers:1,minScale:.1484375,maxScale:.75,inputSizeHeight:192,inputSizeWidth:192,anchorOffsetX:.5,anchorOffsetY:.5,strides:[4],aspectRatios:[1],fixedAnchorSize:!0},B={runtime:"tfjs",modelType:"short",maxFaces:1,detectorModelUrl:C},D={applyExponentialOnBoxSize:!1,flipVertically:!1,ignoreClasses:[],numClasses:1,numBoxes:896,numCoords:16,boxCoordOffset:0,keypointCoordOffset:4,numKeypoints:6,numValuesPerKeypoint:2,sigmoidScore:!0,scoreClippingThresh:100,reverseOutputOrder:!0,xScale:128,yScale:128,hScale:128,wScale:128,minScoreThresh:.5},F={applyExponentialOnBoxSize:!1,flipVertically:!1,ignoreClasses:[],numClasses:1,numBoxes:2304,numCoords:16,boxCoordOffset:0,keypointCoordOffset:4,numKeypoints:6,numValuesPerKeypoint:2,sigmoidScore:!0,scoreClippingThresh:100,reverseOutputOrder:!0,xScale:192,yScale:192,hScale:192,wScale:192,minScoreThresh:.6},A=.3,q={outputTensorSize:{width:128,height:128},keepAspectRatio:!0,outputTensorFloatRange:[-1,1],borderMode:"zero"},E={outputTensorSize:{width:192,height:192},keepAspectRatio:!0,outputTensorFloatRange:[-1,1],borderMode:"zero"};var R=function(){function e(e,t,n){this.detectorModel=t,this.maxFaces=n,"full"===e?(this.imageToTensorConfig=E,this.tensorsToDetectionConfig=F,this.anchors=g(O)):(this.imageToTensorConfig=q,this.tensorsToDetectionConfig=D,this.anchors=g(z));var o=i.tensor1d(this.anchors.map((function(e){return e.width}))),r=i.tensor1d(this.anchors.map((function(e){return e.height}))),a=i.tensor1d(this.anchors.map((function(e){return e.xCenter}))),s=i.tensor1d(this.anchors.map((function(e){return e.yCenter})));this.anchorTensor={x:a,y:s,w:o,h:r}}return e.prototype.dispose=function(){this.detectorModel.dispose(),i.dispose([this.anchorTensor.x,this.anchorTensor.y,this.anchorTensor.w,this.anchorTensor.h])},e.prototype.reset=function(){},e.prototype.detectFaces=function(e,t){return void 0===t&&(t=!1),r(this,void 0,void 0,(function(){var n,o,r,s,u,h,c,l,d,p,m;return a(this,(function(a){switch(a.label){case 0:return null==e?(this.reset(),[2,[]]):(n=i.tidy((function(){var n=i.cast(f(e),"float32");if(t){n=i.squeeze(i.image.flipLeftRight(i.expandDims(n,0)),[0])}return n})),o=x(n,this.imageToTensorConfig),r=o.imageTensor,s=o.transformationMatrix,u=this.detectorModel.execute(r,"Identity:0"),h=w(u),c=h.boxes,[4,b([l=h.logits,c],this.anchorTensor,this.tensorsToDetectionConfig)]);case 1:return 0===(d=a.sent()).length?(i.dispose([n,r,u,l,c]),[2,d]):[4,M(d,this.maxFaces,A)];case 2:return p=a.sent(),m=function(e,t){void 0===e&&(e=[]);var i,n=(i=t,[].concat.apply([],i));return e.forEach((function(e){var t=e.locationData;t.relativeKeypoints.forEach((function(e){var t=v(n,[e.x,e.y]),i=t[0],o=t[1];e.x=i,e.y=o}));var i=t.relativeBoundingBox,o=Number.MAX_VALUE,r=Number.MAX_VALUE,a=Number.MIN_VALUE,s=Number.MIN_VALUE;[[i.xMin,i.yMin],[i.xMin+i.width,i.yMin],[i.xMin+i.width,i.yMin+i.height],[i.xMin,i.yMin+i.height]].forEach((function(e){var t=v(n,e),i=t[0],u=t[1];o=Math.min(o,i),a=Math.max(a,i),r=Math.min(r,u),s=Math.max(s,u)})),t.relativeBoundingBox={xMin:o,xMax:a,yMin:r,yMax:s,width:a-o,height:s-r}})),e}(p,s),i.dispose([n,r,u,l,c]),[2,m]}}))}))},e.prototype.estimateFaces=function(e,t){return r(this,void 0,void 0,(function(){var i,n;return a(this,(function(r){return i=d(e),n=!!t&&t.flipHorizontal,[2,this.detectFaces(e,n).then((function(e){return e.map((function(e){for(var t=e.locationData.relativeKeypoints.map((function(e,t){return o(o({},e),{x:e.x*i.width,y:e.y*i.height,name:s[t]})})),n=e.locationData.relativeBoundingBox,r=0,a=["width","xMax","xMin"];r<a.length;r++){n[a[r]]*=i.width}for(var u=0,h=["height","yMax","yMin"];u<h.length;u++){n[h[u]]*=i.height}return{keypoints:t,box:n}}))}))]}))}))},e}();function L(e){return r(this,void 0,void 0,(function(){var t,i,r;return a(this,(function(a){switch(a.label){case 0:return t=function(e){if(null==e)return o({},B);var t=o({},e);null==t.modelType&&(t.modelType=B.modelType),null==t.maxFaces&&(t.maxFaces=B.maxFaces),null==t.detectorModelUrl&&("full"===t.modelType?t.detectorModelUrl="https://tfhub.dev/mediapipe/tfjs-model/face_detection/full/1":t.detectorModelUrl=C);return t}(e),i="string"==typeof t.detectorModelUrl&&t.detectorModelUrl.indexOf("https://tfhub.dev")>-1,[4,n.loadGraphModel(t.detectorModelUrl,{fromTFHub:i})];case 1:return r=a.sent(),[2,new R(t.modelType,r,t.maxFaces)]}}))}))}(e.SupportedModels||(e.SupportedModels={})).MediaPipeFaceDetector="MediaPipeFaceDetector",e.MediaPipeFaceDetectorMediaPipe=h,e.MediaPipeFaceDetectorTfjs=R,e.createDetector=function(t,i){return r(this,void 0,void 0,(function(){var n,o;return a(this,(function(r){if(t===e.SupportedModels.MediaPipeFaceDetector){if(o=void 0,null!=(n=i)){if("tfjs"===n.runtime)return[2,L(n)];if("mediapipe"===n.runtime)return[2,c(n)];o=n.runtime}throw new Error("Expect modelConfig.runtime to be either 'tfjs' "+"or 'mediapipe', but got ".concat(o))}throw new Error("".concat(t," is not a supported model name."))}))}))},Object.defineProperty(e,"__esModule",{value:!0})}));
