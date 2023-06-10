import Storage from './storage.js';
import path from 'path';
import fs from 'node:fs/promises';
import md5 from 'md5';
import * as helper from './helper.js';

const StorageRoot = "./data";
const SALT = "kdaladder-salt-1324"
const TOKEN_TIMEOUT_SEC = 3600 * 24; // 1 day

class StorageLocalfile extends Storage {
    async _generateGroupID(groupCodeHash) {
        // 寻找不存在的最小正整数作为 ID

        let allGroupIDs = await fs.readdir(StorageRoot).catch(err => []);

        if(allGroupIDs.find(i => i.split("-")[1] == groupCodeHash)) {
            throw new Error("group already exists")
        }

        let id = 0;
        do {
            id ++;
        } while(allGroupIDs.find(i => i.split("-")[0] == id))

        return id;
    }
    async hasPath(groupCodeHash) {
        let allGroupIDs = await fs.readdir(StorageRoot).catch(err => []);

        let name = allGroupIDs.find(i => 
            i == groupCodeHash ||
            i.split("-")[1] == groupCodeHash)
        if(name) {
            return path.join(StorageRoot, name);
        } else {
            throw new Error("group not found")
        }
    }
    async findPath(groupCode) {
        // 根据 code 找到文件夹
        let groupCodeHash = this.groupCodeHash(groupCode);
        return this.hasPath(groupCodeHash);
    }
    async _updateGroupIndex() {
        let allGroupIDs = await (await fs.readdir(StorageRoot).catch(err => [])).filter(i => /^[0-9]{1,}-.{32}$/.test(i));
        await fs.writeFile(path.join(StorageRoot, "index.json"), JSON.stringify(allGroupIDs))
    }
    groupCodeHash(groupCode) {
        // 获取 groupCode 哈希值，用于存储目录命名
        return md5(groupCode + SALT);
    }
    groupCodeToken(groupCodeHashName) {
        let ts = Date.now();
        return `${groupCodeHashName}.${md5(ts + SALT)}.${ts}`
    }
    async verifyGroupCodeToken(token) {
        let tokenSep = token.split(".");
        if(tokenSep.length != 3) {
            throw new Error("token ilegel")
        }
        if(md5(tokenSep[2] + SALT) != tokenSep[1]) {
            throw new Error("token not valid")
        }
        return tokenSep[0];
    }
    async createGroup(groupName, groupCode) {
        if(!groupCode || groupCode.length > 16) {
            throw new Error("password too long (>16)")
        }
        let groupCodeHash = md5(groupCode + SALT)
        let groupID = await this._generateGroupID(groupCodeHash);

        // 把 groupCode 保存在文件夹名字里，公开访问也无法反译
        let groupPath = path.join(StorageRoot, `${groupID}-${groupCodeHash}`);
        let groupNamePath = path.join(groupPath, "name");

        // 创建 group 的目录
        await fs.mkdir(groupPath, {recursive: true})

        await fs.writeFile(groupNamePath, groupName)

        await this._updateGroupIndex();

        return `${groupID}-${groupCodeHash}`
    }
    async updateGroup(oldGroupCode, newGroupCode, groupName) {
        let groupPath = await this.findPath(oldGroupCode);
        let groupID = groupPath.split("-")[0];

        if(newGroupCode && newGroupCode != oldGroupCode) {
            let groupCodeHash = md5(newGroupCode + SALT)
            let newGroupPath = `${groupID}-${groupCodeHash}`;

            if(await this.findPath(newGroupCode).catch(err => false)) {
                throw new Error("group already exists")
            }
            await fs.rename(groupPath, newGroupPath);
            groupPath = newGroupPath
        }

        if(groupName) {
            let groupNamePath = path.join(groupPath, "name");
            await fs.writeFile(groupNamePath, groupName)
        }

        await this._updateGroupIndex();
        
    }
    async getGroup(groupCode) {
        let groupPath = await this.findPath(groupCode);
        let groupName = (await fs.readFile(path.join(groupPath, "name"), {encoding: "utf-8"})).toString();
        return {
            groupName
        }
    }
    async saveMatch(groupCode, matchData, ladderData) {
        let groupPath;
        if(groupCode.length > 16) {
            // is token
            let groupCodeHash = await this.verifyGroupCodeToken(groupCode);
            groupPath = await this.hasPath(groupCodeHash);
        } else {
            groupPath = await this.findPath(groupCode);
        }
        if(matchData) {
            await Promise.all(matchData.map(async data => {
                let matchDataPath = path.join(groupPath, `data-${helper.dateString(new Date(data.beginTime))}.json`);       
            
                // save match
                let dataStr = JSON.stringify(data)
                if(!await fs.access(matchDataPath).catch(err => true)) { // if exist
                    dataStr = "," + dataStr;
                }
                
                // append to file
                await fs.appendFile(matchDataPath, dataStr);
            }))
    
        }

        if(ladderData) {
            await Promise.all(ladderData.map(async data => {
                let ladderDataPath = path.join(groupPath, `ladder-${helper.seasonString(new Date(data.beginTime))}.json`);

                // save ladder
                let dataStr = JSON.stringify(data)
                if(!await fs.access(ladderDataPath).catch(err => true)) { // if exist
                    dataStr = "," + dataStr;
                }

                // append to file
                await fs.appendFile(ladderDataPath, dataStr);
            }))
        }
    }
    async updateMember(groupCode, member) {
        /*
            member = {
                delFaceID: ["..."],
                delPersonID: ["..."],
            }
        */
        let groupPath;
        if(groupCode.length > 16) {
            // is token
            let groupCodeHash = await this.verifyGroupCodeToken(groupCode);
            groupPath = await this.hasPath(groupCodeHash);
        } else {
            groupPath = await this.findPath(groupCode);
        }
        let memberDataPath = path.join(groupPath, `member`)

        // 创建 member 的目录
        await fs.mkdir(memberDataPath, {recursive: true})

        let memberJsonPath = path.join(memberDataPath, `${member.name}.json`)
        if(path.parse(memberJsonPath).name != member.name) {
            throw new Error("member.name 非法")
        }
        
        let storedMember
        if(!await fs.access(memberJsonPath).catch(err => true)) { // if exist
            storedMember = JSON.parse((await fs.readFile(memberJsonPath, {encoding: "utf-8"})).toString());
        } else {
            throw new Error("member 不存在")
        }

        
        //remove image
        member.delFaceID.forEach(async i=>{
            const delIdx = storedMember.faceID.indexOf(i)

            storedMember.faceID.splice(delIdx, 1)

            let removeFile = storedMember.faceImgName.splice(delIdx, 1)[0];
            await fs.unlink(path.join(memberDataPath, removeFile))
        })
        member.delPersonID.forEach(async i=>{
            const delIdx = storedMember.personID.indexOf(i)

            storedMember.personID.splice(delIdx, 1)

            let removeFile = storedMember.personImgName.splice(delIdx, 1)[0];
            await fs.unlink(path.join(memberDataPath, removeFile))
        })

        // save member
        await fs.writeFile(memberJsonPath, JSON.stringify(storedMember))

    }
    async saveMember(groupCode, member) {

        /*
            member = {
                name: "dawei",
                faceID: ["..."],
                personID: ["..."],
                faceImg: [],
                personImg: [],
            }
        */
        let groupPath;
        if(groupCode.length > 16) {
            // is token
            let groupCodeHash = await this.verifyGroupCodeToken(groupCode);
            groupPath = await this.hasPath(groupCodeHash);
        } else {
            groupPath = await this.findPath(groupCode);
        }
        let memberDataPath = path.join(groupPath, `member`)

        // 创建 member 的目录
        await fs.mkdir(memberDataPath, {recursive: true})

        let memberJsonPath = path.join(memberDataPath, `${member.name}.json`)
        if(path.parse(memberJsonPath).name != member.name) {
            throw new Error("member.name 非法")
        }
        if(JSON.stringify(member).length > 1024 * 1024) {
            throw new Error("member 大小非法")
        }
        let memberlistJsonPath = path.join(memberDataPath, `memberlist.json`)
        let storedMemberList = {}
        if(!await fs.access(memberlistJsonPath).catch(err => true)) { // if exist
            storedMemberList = JSON.parse((await fs.readFile(memberlistJsonPath, {encoding: "utf-8"})).toString());
        }

        // 生成图片名字
        // save image
        let now = helper.tickString(new Date());
        member.faceImgName = await Promise.all(member.faceImg.map(async (i,idx)=>{
            let filename = `${member.name}_face_${now}_${idx}.jpg`
            let filepath = path.join(memberDataPath, filename)
            const imageBuffer = Buffer.from(i, 'base64');
            await fs.writeFile(filepath, imageBuffer)
            return filename
        }))
        member.personImgName = await Promise.all(member.personImg.map(async (i,idx)=>{
            let filename = `${member.name}_person_${now}_${idx}.jpg`
            let filepath = path.join(memberDataPath, filename)
            const imageBuffer = Buffer.from(i, 'base64');
            await fs.writeFile(filepath, imageBuffer)
            return filename
        }))
        delete member.faceImg
        delete member.personImg

        let storedMember
        if(!await fs.access(memberJsonPath).catch(err => true)) { // if exist
            storedMember = JSON.parse((await fs.readFile(memberJsonPath, {encoding: "utf-8"})).toString());
            storedMember.faceID = storedMember.faceID.concat(member.faceID)
            storedMember.personID = storedMember.personID.concat(member.personID)
            storedMember.faceImgName = storedMember.faceImgName.concat(member.faceImgName)
            storedMember.personImgName = storedMember.personImgName.concat(member.personImgName)
        } else {
            storedMember = member
        }

        //limit 5, remove image
        if(storedMember.faceID.length > 5) {
            storedMember.faceID.splice(0, storedMember.faceID.length - 5)
            while(storedMember.faceImgName.length > 5) {
                let removeFile = storedMember.faceImgName.shift();
                await fs.unlink(path.join(memberDataPath, removeFile))
            }
        }
        if(storedMember.personID.length > 5) {
            storedMember.personID.splice(0, storedMember.personID.length - 5)
            while(storedMember.personImgName.length > 5) {
                let removeFile = storedMember.personImgName.shift();
                await fs.unlink(path.join(memberDataPath, removeFile))
            }
        }

        // save member
        await fs.writeFile(memberJsonPath, JSON.stringify(storedMember))

        storedMemberList[member.name] = member.name;
        // save memberlist
        await fs.writeFile(memberlistJsonPath, JSON.stringify(storedMemberList))

    }
}

export default StorageLocalfile