"""极小目标审计的 PNG 案例、可离线审核的 HTML 和报告。"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np


def patch(image, region):
    """原像素裁块，展示时才做最近邻放大；不使用锐化或超分辨率。"""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = region
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("展示区域与图像无交集")
    return image[y0:y1, x0:x1].copy()


def save_case(out, row, view, baseline, oracle, oracle_roi, deployed, deployed_roi):
    from nt.tiny import serial
    case = json.loads(json.dumps(row, default=serial))
    path = out / "cases" / row["id"]
    path.mkdir(parents=True, exist_ok=False)
    pts = np.array(row["corners_net"]) / row["scale"]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    extent = max(32, (hi[0] - lo[0]) * 3, (hi[1] - lo[1]) * 4)
    region = np.r_[center - [extent / 2, extent * 3 / 8], center + [extent / 2, extent * 3 / 8]]
    raw_offset = np.array([*view["roi"][:2], *view["roi"][:2]])
    panels = [("raw", "原图像素", patch(view["source_image"], region + raw_offset)),
              ("degraded", "原分辨率增强后", patch(view["native"], region)),
              ("full", "历史网络输入", patch(baseline, region * row["scale"]))]
    for key, title, net, roi in (("oracle", "真值 ROI 网络输入", oracle, oracle_roi),
                                ("deployed", "检测 ROI 覆盖窗口", deployed, deployed_roi)):
        if net is not None:
            x, y, w, _ = roi
            p = patch(net, (region - [x, y, x, y]) * (768 / w))
        else:
            p = None
        panels.append((key, title, p))
    assets, montage = [], np.full((240, 5 * 240, 3), 235, np.uint8)
    for col, (key, title, img) in enumerate(panels):
        asset = {"key": key, "title": title, "path": None, "size": None}
        if img is not None:
            target = path / f"{key}.png"
            if not cv2.imwrite(str(target), img):
                raise OSError(f"无法写入 {target}")
            asset.update(path=target.relative_to(out).as_posix(), size=[img.shape[1], img.shape[0]])
            ratio = min(230 / img.shape[1], 185 / img.shape[0])
            enlarged = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            h, w = enlarged.shape[:2]
            montage[35:35 + h, col * 240 + (240 - w) // 2:col * 240 + (240 - w) // 2 + w] = enlarged
        cv2.putText(montage, key if img is not None else "no proposal", (col * 240 + 8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        assets.append(asset)
    if not cv2.imwrite(str(path / "comparison.png"), montage):
        raise OSError(f"无法写入 {path}")
    case["panels"] = assets
    case["montage"] = (path / "comparison.png").relative_to(out).as_posix()
    case["source_context_native"] = region.tolist()
    case["displayed_detector_roi"] = deployed_roi
    return case


PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>极小目标案例审核</title>
<style>
body{font:16px system-ui,sans-serif;background:#f4f6f8;color:#172331;margin:0}header,main{max-width:1280px;margin:auto;padding:24px}header{position:sticky;top:0;background:#f4f6f8f5;z-index:1;border-bottom:1px solid #ccd4dd}h1{font-size:24px;margin:0 0 12px}p{line-height:1.65}button,select,input{font:inherit;margin:5px;padding:6px}article{background:white;border:1px solid #d2dce5;border-radius:8px;margin:18px 0;padding:18px}h2{font-size:18px;overflow-wrap:anywhere}.panels{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.panel{background:#101820;color:white;text-align:center;padding:8px}.panel img{width:100%;height:185px;object-fit:contain;image-rendering:pixelated}.panel small{display:block;color:#c6d2de}textarea{display:block;box-sizing:border-box;width:100%;height:65px;font:inherit;margin:10px 0}label{display:inline-block;margin:4px 10px 4px 0}.truth{background:#eef3f8;padding:10px;white-space:pre-wrap}#message{color:#476042}a{color:#1859a0}@media(max-width:800px){.panels{grid-template-columns:repeat(2,1fr)}header{position:static}}
</style>
<header><h1>极小目标：原图、退化、缩放与 ROI</h1>
<p>__INTRO__。先隐藏真值判断可辨认性，再查看预测。原始 PNG 裁块仅用最近邻放大。检测 ROI 面板表示覆盖目标的候选窗口，最终预测可能仍来自全图。</p>
<label>案例<select id="group"><option value="all">全部</option><option value="failure">历史失败</option><option value="control">正确对照</option></select></label>
<label>变化<select id="change"><option value="all">全部</option><option value="oracle_recovered">真值 ROI 恢复</option><option value="deploy_recovered">检测 ROI 恢复</option><option value="regressed">检测 ROI 退化</option><option value="pending">尚未审核</option></select></label>
<label><input type="checkbox" id="truth">显示真值与预测</label>
<button id="export">导出审核 JSON</button><label>导入审核<input id="import" type="file" accept="application/json"></label>
<div id="message"></div><div id="count"></div></header><main id="list"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const payload=JSON.parse(document.getElementById('data').textContent), cases=payload.cases;
const key='nt_tiny_review_'+payload.identity;
let review={};try{review=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const el=id=>document.getElementById(id), causes=['原图字形不可辨','缩放损失','增强造成退化','遮挡或倾角','标签或角点问题','图像可辨但模型混淆','暂不能判断'];
const text=(tag,value)=>{const n=document.createElement(tag);n.textContent=value;return n};
function save(){try{localStorage.setItem(key,JSON.stringify(review));el('message').textContent='审核已保存到当前浏览器；请导出 JSON 留档。'}catch(e){el('message').textContent='浏览器无法持久保存，请及时导出 JSON。'}el('count').textContent='显示 '+document.querySelectorAll('article').length+' / '+cases.length+' 个案例；审核 '+cases.filter(c=>state(c.id).status==='reviewed').length+' 个。'}
function state(id){return review[id]||{status:'pending',original_readability:'unknown',causes:[],notes:''}}
function update(id,field,value){review[id]={...state(id),[field]:value,status:'reviewed'};save()}
function prediction(o){const p=o.prediction;return o.state+(p?' / '+payload.class_names[p.cls]+' / conf '+p.conf.toFixed(3):'') }
function draw(){el('list').replaceChildren();const visible=cases.filter(c=>{if(el('group').value!=='all'&&c.group!==el('group').value)return false;switch(el('change').value){case'oracle_recovered':return c.full.state!=='correct'&&c.oracle_roi.state==='correct';case'deploy_recovered':return c.full.state!=='correct'&&c.detector_roi.state==='correct';case'regressed':return c.full.state==='correct'&&c.detector_roi.state!=='correct';case'pending':return state(c.id).status==='pending';default:return true}});
el('count').textContent='显示 '+visible.length+' / '+cases.length+' 个案例；审核 '+cases.filter(c=>state(c.id).status==='reviewed').length+' 个。';
for(const c of visible){const a=document.createElement('article');a.append(text('h2',c.id));a.append(text('p',(c.group==='failure'?'历史失败':'正确对照')+' · 原图宽 '+c.width_orig.toFixed(2)+' px · 网络宽 '+c.width_net.toFixed(2)+' px · 原视图 '+c.historical_roi_mode));const grid=document.createElement('div');grid.className='panels';for(const p of c.panels){const n=document.createElement('div');n.className='panel';n.append(text('div',p.title));if(p.path){const im=document.createElement('img');im.src=p.path;im.loading='lazy';im.alt=p.title;n.append(im);n.append(text('small',p.size.join('×')+' 原始裁块像素'))}else n.append(text('p','没有覆盖该目标的检测 ROI'));grid.append(n)}a.append(grid);
if(el('truth').checked){const t=text('div','真值：'+c.gt_name+'\n全图：'+prediction(c.full)+'\n真值 ROI：'+prediction(c.oracle_roi)+'\n检测 ROI 合并：'+prediction(c.detector_roi));t.className='truth';a.append(t)}
const lab=text('label','原图可辨认性 '), sel=document.createElement('select');for(const [v,label] of [['unknown','未判断'],['readable','可辨认'],['ambiguous','存在歧义'],['unreadable','无法辨认']]){const o=text('option',label);o.value=v;sel.append(o)}sel.value=state(c.id).original_readability;sel.onchange=()=>update(c.id,'original_readability',sel.value);lab.append(sel);a.append(lab);
const box=document.createElement('div');for(const cause of causes){const l=text('label',cause), cb=document.createElement('input');cb.type='checkbox';cb.checked=state(c.id).causes.includes(cause);cb.onchange=()=>{let chosen=[...state(c.id).causes].filter(x=>x!==cause);if(cb.checked)chosen.push(cause);update(c.id,'causes',chosen)};l.prepend(cb);box.append(l)}a.append(box);const note=document.createElement('textarea');note.placeholder='证据或备注；请区分观察与推测';note.value=state(c.id).notes;note.onchange=()=>update(c.id,'notes',note.value);a.append(note);const link=text('a','打开像素对照图');link.href=c.montage;link.target='_blank';a.append(link);el('list').append(a)}}
for(const id of ['group','change','truth'])el(id).onchange=draw;
el('export').onclick=()=>{const data={schema_version:1,experiment_identity:payload.identity,reviews:cases.map(c=>({id:c.id,...state(c.id)}))};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='tiny_reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
el('import').onchange=async e=>{try{const d=JSON.parse(await e.target.files[0].text());if(d.experiment_identity!==payload.identity)throw Error('实验身份不同');const ids=new Set(cases.map(c=>c.id));for(const r of d.reviews){if(!ids.has(r.id)||!Array.isArray(r.causes)||typeof r.notes!=='string'||!['pending','reviewed'].includes(r.status)||!['unknown','readable','ambiguous','unreadable'].includes(r.original_readability))throw Error('审核记录无效')}for(const r of d.reviews)review[r.id]=r;save();draw()}catch(err){el('message').textContent='导入失败：'+err.message}};draw();
</script></html>'''


def write_report(out: Path, summary: dict, cases: list, inputs: dict, targets: list):
    from nt.spec import LabelSpec
    from nt.tiny import serial
    spec = LabelSpec.load()
    payload = {"identity": inputs["weights_sha256"] + ':' + inputs["meta_sha256"] + ':' + out.resolve().name,
               "class_names": {str(c): spec.class_name(c) for c in range(spec.n_classes)}, "cases": cases}
    data = json.dumps(payload, ensure_ascii=False, default=serial).replace("<", "\u003c")
    intro = f"共 {len(cases)} 个案例，{summary['cases'].get('failure', 0)} 个失败和 {summary['cases'].get('control', 0)} 个正确对照"
    (out / "index.html").write_text(PAGE.replace("__INTRO__", html.escape(intro)).replace("__DATA__", data), encoding="utf-8")
    columns = ["id", "group", "image", "gt_name", "width_orig", "width_net", "historical_roi_mode",
               "full", "oracle_roi", "detector_roi", "detector_covered", "review_status"]
    with (out / "targets.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in targets:
            flat = {k: row[k] for k in columns if k in row and k not in ("full", "oracle_roi", "detector_roi")}
            flat.update({r: row[r]["state"] for r in ("full", "oracle_roi", "detector_roi")})
            flat.update(detector_covered=row["detector_roi"]["covered"], review_status=row["review"]["status"])
            writer.writerow(flat)
    lines = ["# 极小目标固定权重 ROI 实验", "",
             "权重固定为 A→B 100 轮的 last.pt，e2e、FP32、置信度 0.2、IoU 0.5；没有训练或调整权重。",
             "查看 [交互案例页](index.html)、[逐目标 CSV](targets.csv)、[完整逐目标记录](targets.jsonl)。", ""]
    names = {"full": "历史输入", "oracle_roi": "真值 ROI（诊断）", "detector_roi": "检测驱动 ROI + 全图"}
    for key, title in (("all_tiny", "固定的全部极小目标"), ("full_frame_only", "其中历史全图目标（主要对照）"),
                       ("historical_roi_only", "历史 ROI 输入目标（补充）"), ("digits_134", "红蓝 1/3/4 数字目标")):
        lines += [f"## {title}", "", "| 路线 | 目标 | 匹配 | 正确 | 召回率 | 匹配后类别准确率 | 检出且正确 / 全部 | 角点 P95 / 原图 px |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for route, m in summary[key].items():
            fmt = lambda v: "—" if v is None else f"{v * 100:.2f}%"
            lines.append(f"| {names[route]} | {m['n']} | {m['hits']} | {m['correct']} | {fmt(m['recall'])} | {fmt(m['class_acc'])} | {fmt(m['end_to_end_correct'])} | {m['corner_p95_orig_px']} |")
        lines.append("")
    lines += ["## 时间和覆盖", "", "| 路线 | 帧数 | P50 ms | P95 ms | P99 ms |", "|---|---:|---:|---:|---:|"]
    for route, t in summary["timing_ms"].items():
        lines.append(f"| {names[route]} | {t['n']} | {t['p50']} | {t['p95']} | {t['p99']} |")
    control = summary["full_merge_only_control"]
    cm = control["metrics"]
    params = inputs["parameters"]
    lines += ["", f"检测 ROI 完整覆盖 {summary['roi_coverage']['covered']}/{summary['roi_coverage']['n']} 个固定目标。",
              f"审计帧上的全图/检测 ROI 合并误检数：{summary['false_positives_on_audit_frames']}。",
              f"全图采用同样去重、不增加 ROI 的补充对照：正确 {cm['correct']}/{cm['n']}，误检 {control['false_positives_on_audit_frames']}。",
              "原始全图到检测 ROI 的误检变化混合了去重与新增预测的作用；应以去重后的全图作为误检成本对照。",
              f"真值 ROI 单目标调用时间：{summary['oracle_per_target_ms']}；不包含搜索目标的成本。",
              "时间为本机 PyTorch FP32、batch=1、预热后同步测量，含缩放、预处理、设备传输、模型及后处理；检测路线包含全图和所有候选 ROI。",
              "不含图片载盘、合成图像重建和报告生成，不代表 TensorRT 或整条自瞄系统时延。", "",
              "## 配对与审核约定", "",
              f"- {summary['frames_pixel_identical']} 张相关输入均按原种子重建原始分辨率图像，缩放后与历史 PNG 逐像素一致。",
              "- 所有路线使用同一份已增强的原始分辨率视图；没有分别重新采样噪声、曝光或模糊。目标集合固定，放大后也不会从分母移除。",
              f"- 历史 ROI 的 {summary['historical_roi_only']['full']['n']} 个目标仅在原先可见区域内比较，单独报告；不把它们称作全图结果。",
              "- 真值 ROI 每个目标单独裁剪，定位由真值提供，仅用于诊断；不报告成可部署搜索能力或总体 AP。",
              f"- 检测路线只用全图预测生成候选，最多 {params['max_rois']} 个 ROI，候选置信度 >={params['seed_conf']}、网络框宽 <{params['proposal_width']} px。窗口最小宽 {params['roi_min']} px，目标框包络乘 {params['roi_multiple']}，保持 4:3。按置信度、IoU 0.5 跨类别合并。",
              "- 覆盖率用真值事后计量；它不参与 ROI 选择。即使没有候选，目标仍保留在分母中，全图输出作为回退。",
              "- 失败判定和控制样本仅依据历史输入，正确对照按类别和尺寸分层并分散图像；全部目标参与正式指标，案例页只展示失败与抽样对照。",
              "- 人工可辨认性及具体失真原因尚待审核，默认 pending；模型恢复不自动证明缩放是唯一原因。页面可保存和导出人工审核。",
              "- 输入为独立静态仿真图，未验证连续跟踪、目标丢失后的重新捕获或真实相机效果。",
              "- 角点误差仅统计成功匹配目标，路线间匹配集合不同。实验是在已检查过的验证集上诊断，不能作为全新独立测试集成绩。", "",
              "复现参数、输入和代码哈希见 inputs.json；完整预测、ROI 窗口与时间见 frames.jsonl。"]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
