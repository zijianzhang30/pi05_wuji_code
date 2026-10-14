from pathlib import Path
import json,collections,math,datetime
N=Path(__file__).resolve().parent
M=json.loads((N/'manifest.json').read_text())
rows=[json.loads(p.read_text()) for p in (N/'results').glob('*.json')]
expected={(t,s) for t in M['evaluation_tasks'] for s in M['seeds']}
actual=[(r['task'],r['seed']) for r in rows]
assert len(rows)==50 and len(set(actual))==50 and set(actual)==expected
assert all(r['valid'] and r['video']['full_decode_passed'] and r['video']['decoded_frames']==r['frames'] for r in rows)
summary={'created_at_bjt':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'checkpoint_step':37000,'valid_episodes':50,'reused_first_seed_episodes':5,'new_episodes':45,'task_count':5,'per_task':{},'scope':'Exploratory five-task10-seed evaluation at checkpoint37000, not final10-epoch benchmark'}
for task in M['evaluation_tasks']:
 rr=[r for r in rows if r['task']==task];k=sum(r['success'] for r in rr);n=len(rr)
 z=1.959963984540054;d=1+z*z/n;mid=(k/n+z*z/(2*n))/d;half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/d
 item={'successes':k,'episodes':n,'rate':k/n,'wilson_95':[max(0,mid-half),min(1,mid+half)],'final_reasons':dict(collections.Counter(r['reason'] for r in rr)),'task_passed_before_geometry':sum(bool(r.get('task_success_before_geometry')) for r in rr),'infrastructure_episode_retries':sum(len(r.get('simulation_infrastructure_errors',[])) for r in rr)}
 details=[]
 for r in rr:
  records=json.loads((Path(r['run'])/'contacts.json').read_text());detail={'seed':r['seed']}
  if task=='piano':detail['events']=records[-1]['events']
  elif task in ['pick_place','stack']:detail['peak_object_contact_N']=max(x['object_contact_N'] for x in records)
  elif task=='produce_sort':detail['peak_food_contact_N']=max(max(x['food_contact_N']) for x in records)
  else:detail.update(max_four_finger_streak=max(x['four_finger_max_streak'] for x in records),max_door_angle_deg=max(x['door_angle_deg'] for x in records))
  details.append(detail)
 item['trace_details']=sorted(details,key=lambda x:x['seed']);summary['per_task'][task]=item
summary['successes']=sum(r['success'] for r in rows);summary['rate']=summary['successes']/50
(N/'RESULT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
labels={'produce_sort':'果蔬分类','stack':'堆叠','pick_place':'抓放','microwave_open':'微波炉','piano':'钢琴'}
lines=['# 37000步：五项任务各10次开发评测','',summary['created_at_bjt'],'','|任务|成功/次数|失败原因统计|','|---|---:|---|']
for t,d in summary['per_task'].items():lines.append(f"|{labels[t]}|{d['successes']}/10|{d['final_reasons']}|")
lines+=['',f"合计{summary['successes']}/50。前次每项seed920101结果计入一次，新运行其余45次。没有按成功与否筛选或重复有效策略失败。",'测试为每任务10个开发种子920101–920110；模型冻结在37000步，约7.23等效帧epoch。不能当作最终10epoch模型性能，也不能外推到真实硬件。','所有有效episode视频全帧解码及帧数检查通过；计入成功必须另外通过原任务、接触与记录帧几何标准。','逐种子失败细节、Wilson区间和基础设施重试见RESULT_SUMMARY.json。测试后自动从测试前最新完整训练检查点恢复，原10epoch目标保留。']
(N/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='per_task'},ensure_ascii=False))
for t,x in summary['per_task'].items():print(t,x['successes'],x['episodes'],x['final_reasons'])
