import Storage from './storage.js';
import path from 'path';
import fs from 'node:fs/promises';
import md5 from 'md5';
import * as helper from './helper.js';

const StorageRoot = "./data";
const SALT = "kdaladder-salt-1324"

class StorageLocalfile extends Storage {
    async _generateGroupID(groupCodeHash) {
        // 寻找不存在的最小正整数作为 ID

        let allGroupIDs = await fs.readdir(StorageRoot).catch(err => []);

        if(allGroupIDs.find(i => i.split("-")[1] == groupCodeHash)) {
            throw "group already exists"
        }

        let id = 0;
        do {
            id ++;
        } while(allGroupIDs.find(i => i.split("-")[0] == id))

        return id;
    }
    async _findPath(groupCode) {
        // 根据 code 找到文件夹
        let groupCodeHash = this.groupCodeHash(groupCode);
        let allGroupIDs = await fs.readdir(StorageRoot).catch(err => []);

        let name = allGroupIDs.find(i => i.split("-")[1] == groupCodeHash)
        if(name) {
            return path.join(StorageRoot, name);
        } else {
            throw "group not found"
        }
    }
    async _updateGroupIndex() {
        let allGroupIDs = await (await fs.readdir(StorageRoot).catch(err => [])).filter(i => /^[0-9]{1,}-.{32}$/.test(i));
        await fs.writeFile(path.join(StorageRoot, "index.json"), JSON.stringify(allGroupIDs))
    }
    groupCodeHash(groupCode) {
        // 获取 groupCode 哈希值，用于存储目录命名
        return md5(groupCode + SALT);
    }
    async createGroup(groupName, groupCode) {
        let groupCodeHash = md5(groupCode + SALT)
        let groupID = await this._generateGroupID(groupCodeHash);

        // 把 groupCode 保存在文件夹名字里，公开访问也无法反译
        let groupPath = path.join(StorageRoot, `${groupID}-${groupCodeHash}`);
        let groupNamePath = path.join(groupPath, "name");

        // 创建 group 的目录
        await fs.mkdir(groupPath, {recursive: true})

        await fs.writeFile(groupNamePath, groupName)

        await this._updateGroupIndex();

        return groupID
    }
    async updateGroup(oldGroupCode, newGroupCode, groupName) {
        let groupPath = await this._findPath(oldGroupCode);
        let groupID = groupPath.split("-")[0];

        let groupCodeHash = md5(newGroupCode + SALT)
        let newGroupPath = `${groupID}-${groupCodeHash}`;

        if(await this._findPath(newGroupCode).catch(err => false)) {
            throw "group already exists"
        }
        await fs.rename(groupPath, newGroupPath);

        if(groupName) {
            let groupNamePath = path.join(newGroupPath, "name");
            await fs.writeFile(groupNamePath, groupName)
        }

        await this._updateGroupIndex();
        
    }
    async saveMatch(groupCode, matchData, ladderData) {
        let groupPath = await this._findPath(groupCode);
        let matchDataPath = path.join(groupPath, `data-${helper.dateString(new Date(matchData.beginTime))}.json`);
        let ladderDataPath = path.join(groupPath, `ladder-${helper.seasonString(new Date(ladderData.beginTime))}.json`);
        // save match
        let dataStr = JSON.stringify(matchData)
        if(!await fs.access(matchDataPath).catch(err => true)) { // if exist
            dataStr = "," + dataStr;
        }

        await fs.appendFile(matchDataPath, dataStr);
        
        // save ladder
        dataStr = JSON.stringify(ladderData)
        if(!await fs.access(ladderDataPath).catch(err => true)) { // if exist
            dataStr = "," + dataStr;
        }

        // append to file
        await fs.appendFile(ladderDataPath, dataStr);
    }
}

export default StorageLocalfile