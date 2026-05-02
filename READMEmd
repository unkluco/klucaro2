# Caro ResNet Training README

Tài liệu này giải thích chi tiết notebook `train.ipynb`: model nhận input gì, chơi game như thế nào, batch hoạt động ra sao, snapshot opponent là gì, reward được tính thế nào, và cuối cùng loss được gom lại ra sao để train model.

---

## 1. Mục tiêu tổng quát

Notebook này đang train một model policy cho game caro trên bàn `15 x 15`.

Model không trực tiếp dự đoán thắng/thua. Model nhận trạng thái bàn cờ và xuất ra một ma trận score kích thước `15 x 15`:

```text
scores[row, col] = model nghĩ ô (row, col) đáng đánh đến mức nào
```

Sau đó hệ thống dùng luật caro để:

1. mask các ô không hợp lệ;
2. chọn ô hợp lệ có score cao nhất;
3. cập nhật bàn cờ;
4. tạo dữ liệu training từ các nước learner đã đi;
5. tính loss dựa trên reward heuristic cục bộ;
6. cộng thêm ảnh hưởng của vài nước tương lai của learner trong cùng ván.

Workflow hiện tại là:

```text
learner model đang train
        vs
snapshot frozen của learner tại đầu train step
```

Quan trọng: snapshot là bản copy đóng băng, không nhận gradient.

---

## 2. Biểu diễn bàn cờ

Mỗi board có shape:

```python
(3, BOARD_SIZE, BOARD_SIZE)
```

Với `BOARD_SIZE = 15`, tức là:

```python
(3, 15, 15)
```

Ba channel có ý nghĩa:

```text
channel 0: quân của người sắp đi
channel 1: quân của đối thủ
channel 2: ô hợp lệ, 1 là còn đánh được, 0 là đã bị chiếm
```

Ví dụ:

```python
board[0, 7, 7] = 1  # người sắp đi đang có quân ở (7, 7)
board[1, 7, 8] = 1  # đối thủ có quân ở (7, 8)
board[2, 7, 7] = 0  # ô (7, 7) không còn hợp lệ
board[2, 7, 8] = 0  # ô (7, 8) không còn hợp lệ
```

### 2.1. Góc nhìn tương đối

Board luôn được biểu diễn từ góc nhìn của người sắp đi.

Nghĩa là `board[0]` không cố định là đen hay trắng. Nó luôn là:

```text
quân của người chuẩn bị đánh nước tiếp theo
```

Sau mỗi nước đi, nếu game chưa kết thúc, board được đảo channel:

```python
board_next = torch.stack([enemy_after, our_after, legal_after], dim=0)
```

Vì sao phải đảo?

Giả sử lượt hiện tại là A:

```text
board[0] = quân A
board[1] = quân B
```

A đánh xong, tới lượt B. Từ góc nhìn mới:

```text
board_next[0] = quân B
board_next[1] = quân A
```

Cách này giúp model luôn nhìn board cùng một quy ước: channel 0 là mình, channel 1 là đối thủ.

---

## 3. Model `CaroResNet`

Model chính là một ResNet nhỏ gồm:

1. stem convolution;
2. nhiều residual block;
3. policy head xuất score cho từng ô.

Input:

```python
(B, 3, 15, 15)
```

Output:

```python
(B, 15, 15)
```

Trong đó:

```text
B = batch size, số board được xử lý cùng lúc
```

Model không tự chọn nước. Model chỉ xuất score. Việc chọn nước hợp lệ được xử lý bởi hàm `select_legal_move`.

---

## 4. Legal mask và chọn nước

Hàm liên quan:

```python
select_legal_move(scores_t, legal_mask)
```

Trong caro, model không được đánh vào ô đã có quân. Do đó ta dùng legal mask:

```python
legal_mask = board_t[2] > 0.5
```

Sau đó score của ô không hợp lệ bị đổi thành số rất âm:

```python
masked_scores = scores_t.masked_fill(~legal_mask, -1e9)
```

Rồi chọn nước:

```python
move_idx = torch.argmax(masked_scores).item()
```

### 4.1. Model có học ô không hợp lệ không?

Hiện tại không train riêng để model giảm score ô không hợp lệ.

Thiết kế hiện tại coi legal mask là một phần của luật môi trường:

```text
model học: trong các ô hợp lệ, ô nào tốt hơn
legal mask lo: ô nào được phép đánh
```

Đây là cách thường dùng trong game AI. Luật hợp lệ không cần model tự học lại từ đầu.

---

## 5. Cập nhật board sau một nước

Hàm liên quan:

```python
step_board_from_model_output(board_t, scores_t)
```

Luồng xử lý:

1. lấy legal mask từ `board_t[2]`;
2. chọn nước hợp lệ bằng `select_legal_move`;
3. đặt quân vào `our_after`;
4. cập nhật `legal_after`;
5. kiểm tra thắng;
6. kiểm tra hòa;
7. nếu chưa kết thúc thì đảo góc nhìn cho người tiếp theo.

Kết quả trả về:

```python
(board_next, "continue")
(board_after, "win")
(board_after, "draw")
(board_t, "invalid")
```

### 5.1. Vì sao terminal board không đảo góc nhìn?

Nếu một nước tạo thắng hoặc hòa, ta giữ board theo góc nhìn người vừa đi:

```python
board_after_current_view = torch.stack([our_after, enemy_after, legal_after], dim=0)
```

Lý do: để visualization thấy đúng nước cuối vừa được đánh.

Nếu đảo ngay sang người tiếp theo trong terminal state thì dễ bị nhầm màu hoặc mất ý nghĩa “người vừa đi”.

---

## 6. Snapshot opponent

Hàm liên quan:

```python
clone_frozen_snapshot(model)
```

Mỗi train step tạo một bản copy đóng băng của learner:

```python
snapshot = copy.deepcopy(model)
snapshot.eval()
for parameter in snapshot.parameters():
    parameter.requires_grad_(False)
```

Snapshot dùng để làm đối thủ trong rollout.

### 6.1. Vì sao cần snapshot?

Nếu learner tự đấu với chính object model đang train, cả hai phía đều thay đổi cùng graph, dễ gây rối gradient và có thể học mẹo để giảm loss.

Snapshot giúp tách vai trò:

```text
learner: model đang update
snapshot: đối thủ cố định trong train step đó
```

### 6.2. Snapshot có nhận gradient không?

Không.

Khi snapshot đi nước, code chạy trong:

```python
with torch.no_grad():
    active_scores = snapshot_model(active_boards)
```

Vì vậy snapshot chỉ tạo nước đi, không đóng góp loss, không cập nhật trọng số.

---

## 7. Batch rollout nhiều ván song song

Hàm chính:

```python
rollout_batch_vs_snapshot_opponent(
    learner_model,
    snapshot_model,
    batch_size=TRAIN_BATCH_SIZE,
    max_moves=TRAIN_MAX_MOVES,
)
```

Batch ở đây nghĩa là nhiều ván đang chạy song song.

Nếu `TRAIN_BATCH_SIZE = 8`, ta có:

```python
boards.shape == (8, 3, 15, 15)
```

Mỗi index batch là một game slot:

```text
boards[0] = ván 0
boards[1] = ván 1
...
boards[7] = ván 7
```

---

## 8. `done_mask`: ván nào xong thì đứng yên

Trong batch, có ván kết thúc sớm, có ván còn chơi tiếp. Notebook dùng `done`:

```python
done = torch.zeros(batch_size, dtype=torch.bool, device=device)
```

Ở mỗi move:

```python
active_indices = torch.nonzero(~done, as_tuple=False).flatten()
```

Chỉ game chưa done mới được xử lý.

Nếu một game kết thúc:

```python
done[game_i] = True
terminal_reasons[game_i] = reason
```

Board của game đó đứng yên và không sinh thêm move.

### 8.1. Batch size có bị giảm không?

Không.

Tensor `boards` vẫn có shape cố định:

```python
(B, 3, 15, 15)
```

Chỉ số game active giảm dần qua `done_mask`.

---

## 9. Learner đi trước hay đi sau

Code chia nửa batch:

```python
learner_is_black = torch.arange(batch_size, device=device) % 2 == 0
current_player_is_black = torch.ones(batch_size, dtype=torch.bool, device=device)
```

Ý nghĩa:

```text
game index chẵn: learner là đen, đi trước
game index lẻ: learner là trắng, đi sau
```

Vì caro đen đi trước, ban đầu:

```python
current_player_is_black = True
```

Tới lượt learner khi:

```python
current_player_is_black[game_i] == learner_is_black[game_i]
```

Nếu không phải lượt learner thì là lượt snapshot.

---

## 10. Chỉ lưu move của learner

Trong rollout, chỉ nhánh learner append dữ liệu train:

```python
learner_boards.append(board_before.clone())
learner_scores.append(scores_t)
learner_game_ids.append(game_i)
```

Nhánh snapshot chỉ cập nhật board, không lưu loss.

Điều này đúng với mục tiêu hiện tại:

```text
chỉ train learner
không train đối thủ
không đưa loss của đối thủ vào future loss
```

---

## 11. Vì sao cần `learner_game_ids`?

Learner moves được lưu thành list phẳng.

Ví dụ batch 3 game, thứ tự learner move có thể là:

```python
game_ids = [0, 2, 1, 0, 2, 1, 0, 2]
```

Nếu ta tính future loss trực tiếp trên list này, thì move tiếp theo có thể thuộc game khác.

Ví dụ:

```text
move 0: game 0
move 1: game 2
```

Nếu cộng future từ move 0 sang move 1 thì sai, vì đó không phải tương lai của cùng ván.

Do đó mỗi learner move lưu thêm `game_i`:

```python
learner_game_ids.append(game_i)
```

Sau đó loss được gom riêng từng game.

---

## 12. Reward heuristic

Reward được tính bởi:

```python
window_reward(board_t, row, col)
build_reward_map(board_t)
```

Ý tưởng: giả sử learner đánh vào ô `(row, col)`, ta xét các cửa sổ 5 ô đi qua vị trí đó.

Bốn hướng:

```python
directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
```

Tương ứng:

```text
dọc
ngang
chéo xuống
chéo lên
```

Một ô có thể nằm ở 5 vị trí khác nhau trong một cửa sổ 5 ô, nên với mỗi hướng có `offset in range(5)`.

---

## 13. Cách tính `window_reward`

Với mỗi cửa sổ 5 ô:

1. Nếu cửa sổ vượt ra ngoài bàn cờ thì bỏ qua.
2. Nếu cửa sổ có cả quân ta và quân địch thì bỏ qua.
3. Nếu cửa sổ chỉ có quân ta, thưởng tấn công.
4. Nếu cửa sổ chỉ có quân địch, thưởng phòng thủ.
5. Nếu đánh vào đó tạo 5 quân thì thưởng rất lớn.
6. Nếu chặn đối thủ 4 quân thì cũng thưởng gần như thắng.

Cụ thể:

```python
if our_count == 4:
    reward += win_reward
elif enemy_count == 4:
    reward += win_reward * 0.9
elif our_count > 0:
    reward += alpha ** our_count
elif enemy_count > 0:
    reward += 0.8 * (alpha ** enemy_count)
```

Với mặc định:

```python
REWARD_ALPHA = 8.0
WIN_REWARD = alpha ** 6
```

Ví dụ:

```text
1 quân gần đó: 8^1 = 8
2 quân gần đó: 8^2 = 64
3 quân gần đó: 8^3 = 512
4 quân ta + ô đang xét: win_reward = 8^6
4 quân địch cần chặn: 0.9 * 8^6
```

---

## 14. `build_reward_map`

Hàm:

```python
build_reward_map(board_t)
```

Tạo ma trận:

```python
reward_map.shape == (15, 15)
```

Mỗi ô hợp lệ nhận reward từ `window_reward`.

Ô không hợp lệ giữ reward `0`, vì ô không hợp lệ sẽ bị mask khỏi loss.

---

## 15. Normalize reward

Reward thô có thể rất lớn, ví dụ:

```python
8 ** 6 = 262144
```

Nếu đưa trực tiếp vào softmax/loss, gradient có thể rất mạnh và training không ổn định.

Do đó notebook chuẩn hóa reward:

```python
normalize_reward_map(reward_map, legal_mask)
```

Chỉ các ô hợp lệ được dùng để tính mean/std:

```python
legal_rewards = reward_map[legal_mask]
mean = legal_rewards.mean()
std = legal_rewards.std(unbiased=False).clamp_min(eps)
```

Sau đó:

```python
normalized[legal_mask] = (legal_rewards - mean) / std
```

Cuối cùng clamp:

```python
normalized.clamp(min=-5.0, max=5.0)
```

---

## 16. Local step loss

Hàm:

```python
local_step_loss(scores_t, board_t)
```

Input:

```python
scores_t.shape == (15, 15)
board_t.shape == (3, 15, 15)
```

Luồng:

1. lấy legal mask;
2. tính reward map;
3. normalize reward;
4. mask scores ô không hợp lệ;
5. tạo target policy từ normalized reward;
6. tính cross entropy mềm giữa target policy và log policy của model.

Code chính:

```python
legal_mask = board_t[2] > 0.5
reward_map = build_reward_map(board_t)
normalized_reward = normalize_reward_map(reward_map, legal_mask)

masked_scores = scores_t.masked_fill(~legal_mask, -1e9)

target_policy = torch.softmax(
    normalized_reward.masked_fill(~legal_mask, -1e9).flatten(),
    dim=0,
)
log_policy = torch.log_softmax(masked_scores.flatten(), dim=0)

loss = -(target_policy * log_policy).sum()
```

### 16.1. Ý nghĩa

Model đang học:

```text
phân phối xác suất của model trên các ô hợp lệ
nên giống phân phối target sinh ra từ reward heuristic
```

Ô reward cao sẽ có target probability cao hơn.

---

## 17. Vì sao dùng soft target thay vì chọn một ô tốt nhất?

Nếu chỉ chọn ô reward cao nhất làm target, training sẽ rất cứng:

```text
ô tốt nhất: 1.0
mọi ô khác: 0.0
```

Nhưng caro thường có nhiều nước tương đối tốt. Soft target giúp:

```text
nước rất tốt: xác suất cao
nước khá tốt: vẫn có xác suất
nước tệ: xác suất thấp
```

Điều này làm gradient mượt hơn.

---

## 18. Compute local losses

Hàm:

```python
compute_local_losses(scores, boards)
```

Input là danh sách các board/scores của learner.

Ví dụ:

```python
scores.shape == (N, 15, 15)
boards.shape == (N, 3, 15, 15)
```

Trong đó:

```text
N = tổng số learner moves được lưu trong batch rollout
```

Hàm trả:

```python
local_losses.shape == (N,)
```

Mỗi phần tử là local loss của một learner move.

---

## 19. Future loss của learner

Hàm:

```python
learner_future_loss(local_losses)
```

Hàm này chỉ dùng cho một game riêng lẻ.

Nếu một ván learner có các local loss:

```text
L0, L1, L2, L3, ...
```

Thì loss tại bước `t` là:

```text
true_loss[t] = L[t]
             + future_weight * gamma^1 * L[t+1]
             + future_weight * gamma^2 * L[t+2]
             + ...
             đến tối đa m bước tương lai
```

Công thức:

```python
loss_t = local_losses[t]
for i in range(1, m + 1):
    future_t = t + i
    if future_t >= num_moves:
        break
    loss_t = loss_t + future_weight * (gamma ** i) * local_losses[future_t]
```

### 19.1. Vì sao không đổi dấu?

Vì trajectory này chỉ chứa bước của learner.

Các bước của snapshot đã bị bỏ khỏi loss.

Do đó:

```text
local_losses[t+1] là bước learner tiếp theo
không phải bước đối thủ
```

Nên không cần đổi dấu theo lượt.

---

## 20. Future loss theo từng game

Hàm:

```python
learner_future_loss_by_game(local_losses, game_ids, batch_size)
```

Vì batch rollout lưu learner moves thành list phẳng, ta phải gom lại theo `game_ids`.

Pseudo-code:

```python
for game_i in range(batch_size):
    mask = game_ids == game_i
    game_losses = local_losses[mask]
    game_loss = learner_future_loss(game_losses)
    per_game_losses.append(game_loss)

final_loss = mean(per_game_losses)
```

Điều này đảm bảo:

```text
future loss chỉ cộng các bước learner tiếp theo trong cùng một ván
```

Không cộng nhầm qua game khác.

---

## 21. Train step

Hàm:

```python
train_step_batch_vs_snapshot(model, optimizer)
```

Luồng đầy đủ:

1. đặt learner model ở train mode;
2. tạo snapshot frozen;
3. rollout nhiều ván song song;
4. lấy learner boards/scores;
5. tính local losses;
6. tính future loss theo từng game;
7. backprop;
8. optimizer step;
9. trả thống kê.

Pseudo-code:

```python
model.train()
snapshot_model = clone_frozen_snapshot(model)

rollout = rollout_batch_vs_snapshot_opponent(
    learner_model=model,
    snapshot_model=snapshot_model,
)

local_losses = compute_local_losses(
    rollout["learner_scores"],
    rollout["learner_boards"],
)

loss = learner_future_loss_by_game(
    local_losses,
    rollout["learner_game_ids"],
    batch_size=batch_size,
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 22. Train loop

Hàm được chọn trong notebook:

```python
train_loop_batch_vs_snapshot(...)
```

Mỗi step gọi:

```python
train_step_batch_vs_snapshot(...)
```

Và in thống kê:

```text
step=001 | loss=... | avg_local=... | learner_moves=... | done=... | W/L/D=... | unfinished=... | avg_len=...
```

Ý nghĩa:

```text
loss: loss cuối dùng để backward
avg_local: trung bình local_step_loss trước khi cộng future
learner_moves: tổng số nước learner đã đi trong rollout batch
done: số game kết thúc trong max_moves
W/L/D: learner thắng / snapshot thắng / hòa
unfinished: số game chưa kết thúc khi chạm max_moves
avg_len: độ dài trung bình ván trong batch
```

---

## 23. Vì sao có `unfinished_games`?

`TRAIN_MAX_MOVES` hiện là:

```python
TRAIN_MAX_MOVES = 100
```

Nhưng bàn `15 x 15` có tối đa 225 nước.

Nếu một ván chưa thắng/hòa trong 100 nước, rollout dừng vì giới hạn train:

```text
reason = max_moves
```

Đây là truncation để train nhanh hơn, không phải hòa thật.

Nếu muốn mô phỏng ván đầy đủ hơn, có thể tăng:

```python
TRAIN_MAX_MOVES = BOARD_SIZE * BOARD_SIZE
```

---

## 24. Visualization

Hàm:

```python
play_two_models_visual(model_black, model_white=None)
```

Nếu `model_white=None`, cùng model được dùng cho hai bên để xem policy đánh ra sao.

Hàm này không dùng để train. Nó dùng để quan sát.

Model được đặt ở eval mode:

```python
model_black.eval()
model_white.eval()
```

Và chạy trong:

```python
with torch.no_grad():
```

---

## 25. Những điểm quan trọng cần nhớ

### 25.1. Model không tự học luật ô hợp lệ

Legal mask xử lý luật:

```text
ô đã có quân thì không được chọn
```

Model học chiến thuật trong tập ô hợp lệ.

### 25.2. Snapshot không học

Snapshot chỉ là đối thủ cố định trong một train step.

Gradient chỉ cập nhật learner.

### 25.3. Chỉ learner moves có loss

Snapshot moves không đi vào loss.

### 25.4. Future loss phải tách theo game

Không được cộng future loss trên list phẳng toàn batch.

Phải dùng `learner_game_ids`.

### 25.5. Reward hiện tại là heuristic

Model đang học theo reward cục bộ dựa trên cửa sổ 5 ô, chưa có value head và chưa dùng kết quả thắng/thua làm target trực tiếp.

---

## 26. Hạn chế hiện tại

### 26.1. Snapshot quá gần learner

Hiện mỗi train step tạo snapshot ngay từ learner hiện tại. Snapshot này rất gần learner.

Sau này có thể cải thiện bằng snapshot pool:

```text
snapshot cũ 100 step trước
snapshot cũ 500 step trước
best checkpoint
heuristic bot
random bot
```

### 26.2. Không có outcome loss

Hiện loss chưa dùng kết quả cuối ván kiểu:

```text
win = +1
loss = -1
draw = 0
```

Nó chỉ dùng local heuristic + future local losses.

### 26.3. Batch done slots đứng yên

Khi game kết thúc sớm, slot đó không reset game mới. Đây là thiết kế đơn giản, dễ hiểu.

Sau này có thể tối ưu bằng worker-style rollout:

```text
game nào xong thì reset slot đó thành ván mới
```

### 26.4. BatchNorm có thể dao động

Model dùng `BatchNorm2d`. Batch rollout hiện có batch size hữu ích thay đổi theo số game active.

Nếu training dao động, có thể cân nhắc:

```text
GroupNorm
LayerNorm
freeze BatchNorm sau warmup
```

---

## 27. Luồng training tóm tắt

```text
train_loop_batch_vs_snapshot
    |
    v
train_step_batch_vs_snapshot
    |
    |-- clone_frozen_snapshot(model)
    |
    |-- rollout_batch_vs_snapshot_opponent
    |       |
    |       |-- tạo B board rỗng
    |       |-- chia learner đi trước/đi sau
    |       |-- lặp move_number
    |       |-- nếu lượt learner:
    |       |       model(board)
    |       |       chọn legal move
    |       |       lưu board/scores/game_id
    |       |       update board
    |       |
    |       |-- nếu lượt snapshot:
    |       |       snapshot(board) trong no_grad
    |       |       chọn legal move
    |       |       update board
    |       |
    |       |-- game kết thúc thì done=True
    |
    |-- compute_local_losses
    |
    |-- learner_future_loss_by_game
    |
    |-- loss.backward()
    |
    |-- optimizer.step()
```

---

## 28. Công thức loss tổng quát

Với mỗi learner move `k`:

```text
local_loss[k] = CE(target_policy_from_reward, model_policy)
```

Với mỗi game `g`, giả sử learner có local losses:

```text
L_g[0], L_g[1], ..., L_g[n-1]
```

Loss tại bước `t` trong game đó:

```text
T_g[t] = L_g[t] + sum_{i=1..m} future_weight * gamma^i * L_g[t+i]
```

Nếu `t+i` vượt quá số learner moves trong game thì dừng.

Loss của game:

```text
GameLoss_g = mean_t(T_g[t])
```

Loss cuối cùng của batch:

```text
BatchLoss = mean_g(GameLoss_g)
```

Chỉ tính các game có ít nhất một learner move.

---

## 29. Các hàm chính cần đọc theo thứ tự

Nếu muốn hiểu code từ dễ đến khó, nên đọc theo thứ tự:

1. `CaroResNet`
2. `select_legal_move`
3. `step_board_from_model_output`
4. `window_reward`
5. `build_reward_map`
6. `local_step_loss`
7. `clone_frozen_snapshot`
8. `rollout_batch_vs_snapshot_opponent`
9. `learner_future_loss_by_game`
10. `train_step_batch_vs_snapshot`
11. `train_loop_batch_vs_snapshot`

---

## 30. Kết luận

Notebook hiện tại train theo cơ chế:

```text
learner policy
  đấu nhiều ván song song
  với snapshot frozen
  chỉ học từ các nước learner đã đi
  dùng reward heuristic cục bộ
  cộng thêm ảnh hưởng m bước learner tương lai trong cùng game
```

Legal move được xử lý bằng mask, không bằng loss riêng cho ô lỗi.

Điểm quan trọng nhất của thiết kế hiện tại là:

```text
future loss phải được tính theo từng game, không tính trên list phẳng toàn batch
```

Vì vậy `learner_game_ids` là phần bắt buộc để loss đúng.
