import os
import os.path as osp
import shutil
import tempfile

import pytest

from utils.config import Config

data_path = osp.join(osp.dirname(__file__), "data")


def test_construct():
    cfg = Config()
    assert cfg.filename is None
    assert cfg.text == ""
    assert len(cfg) == 0
    assert cfg._cfg_dict == {}

    with pytest.raises(TypeError):
        Config([0, 1])

    cfg_dict = dict(item1=[1, 2], item2=dict(a=0), item3=True, item4="test")
    # test a.py
    cfg_file = osp.join(data_path, "a.py")
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, "r").read()
    # assert cfg.dump() == cfg.pretty_text
    # with tempfile.TemporaryDirectory() as temp_config_dir:
    #     dump_file = osp.join(temp_config_dir, "a.py")
    #     cfg.dump(dump_file)
    #     assert cfg.dump() == open(dump_file, "r").read()
    #     assert Config.fromfile(dump_file)


def test_fromfile():
    for filename in ["a.py"]:
        cfg_file = osp.join(data_path, filename)
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == osp.abspath(osp.expanduser(cfg_file)) + "\n" + open(cfg_file, "r").read()

    # test custom_imports for Config.fromfile
    # cfg_file = osp.join(data_path, 'q.py')
    # imported_file = osp.join(data_path, 'r.py')
    # target_pkg = osp.join(osp.dirname(__file__), 'r.py')

    # Since the imported config will be regarded as a tmp file
    # it should be copied to the directory at the same level
    # shutil.copy(imported_file, target_pkg)
    Config.fromfile(cfg_file)

    with pytest.raises(FileNotFoundError):
        Config.fromfile("no_such_file.py")


def test_merge_from_base():
    cfg_file = osp.join(data_path, "d.py")
    cfg = Config.fromfile(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    base_cfg_file = osp.join(data_path, "base.py")
    merge_text = osp.abspath(osp.expanduser(base_cfg_file)) + "\n" + open(base_cfg_file, "r").read()
    merge_text += "\n" + osp.abspath(osp.expanduser(cfg_file)) + "\n" + open(cfg_file, "r").read()
    assert cfg.text == merge_text
    assert cfg.dict["item1"] == [2, 3]
    assert cfg.dict["item2"]["a"] == 1
    assert cfg.dict["item3"] is False
    assert cfg.dict["item4"] == "test_base"

    with pytest.raises(TypeError):
        Config.fromfile(osp.join(data_path, "e.py"))
