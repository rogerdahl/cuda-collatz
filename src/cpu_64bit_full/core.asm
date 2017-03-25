; 128 bit x64 Collatz Delay Record Calculator (C)2008 Roger Dahl

bits 64
default rel

global collatz_calc

section .bss

  ; args
  n_base_ptr      resq 1
  delay_high_ptr  resq 1
  sieve_buf_ptr   resq 1
  sieve_size      resq 1

  ; vars
  current_n       resq 1

  
section .code

  ; constants
  four dq 4

collatz_calc:

  ; our allocated stack space = 0x40
  ; spill space allocated by caller = 0x20
  ; push of rbp = 0x08
  ; push by call = 0x08
  %assign off 0x70

  %assign tail_buf_off         off + 0x00
  %assign tail_size_off        off + 0x08

  %assign step_bits_off        off + 0x10
  %assign d_high_bit_off       off + 0x18
  %assign step_table_c_d_off   off + 0x20
  %assign step_table_exp3_off  off + 0x28

  %assign total_loops_0_off    off + 0x30
  %assign total_loops_1_off    off + 0x38
  
  %define b32 dword
  %define b64 qword

  
  ; rax tmp
  ; rbx tmp
  ; rcx tmp
  ; rdx tmp
  %define step_bits       rsi
  %define tail_size       rdi
  %define step_mask       r8
  %define c_d             r9
  %define n_0             r10
  %define n_1             r11
  %define a_0             r12
  %define a_1             r13
  %define delay           r14
  %define delay_high      r15
  ; rbp   frame pointer
  ; rsp   stack pointer

PROC_FRAME		collatz
  db			0x48				; emit a REX prefix to enable hot-patching

  push		rbp					; save prospective frame pointer
  [pushreg	rbp]				; create unwind data for this rbp register push
  sub			rsp, 0x40			; allocate stack space (0x40 = 64 = 8 x 64 bit registers)
  [allocstack 0x40]				; create unwind data for this stack allocation
  lea			rbp, [rsp]; no bias
  [setframe	rbp, 0x00]			; create unwind data for a frame register in rbp
  mov			[rbp + 0x00], rsi	; save rsi
  [savereg	rsi, 0x00]			; create unwind data for a save of rsi
  mov			[rbp + 0x08], rdi
  [savereg	rdi, 0x08]
  mov			[rbp + 0x10], rbx
  [savereg	rbx, 0x10]
  mov			[rbp + 0x18], rcx
  [savereg	rcx, 0x18]
  mov			[rbp + 0x20], r12
  [savereg	r12, 0x20]
  mov			[rbp + 0x28], r13
  [savereg	r13, 0x28]
  mov			[rbp + 0x30], r14
  [savereg	r14, 0x30]
  mov			[rbp + 0x38], r15
  [savereg	r15, 0x38]
  [endprolog]

  ; Store the params that were passed in registers.
  mov [n_base_ptr], rcx
  mov [delay_high_ptr], rdx
  mov [sieve_buf_ptr], r8
  mov [sieve_size], r9

  ; Prepare constants.
  
  ; step_bits
  mov step_bits, [rbp + step_bits_off]  

  ; step_mask  
  mov rax, 1
  mov rcx, step_bits
  shl rax, cl
  mov step_mask, rax
  sub step_mask, 1

  ; tail_size
  mov tail_size, [rbp + tail_size_off]

  ; delay_high
  mov rax, [delay_high_ptr]
  mov delay_high, [rax]

  ; n_base
  mov rax, [n_base_ptr]
  movq mm0, [rax]
  
  ; sieve_buf_ptr (we adjust this as we go).
  movq mm1, [sieve_buf_ptr]
  
  ; total_loops_ptrs
  movq mm2, [rbp + total_loops_0_off]
  movq mm3, [rbp + total_loops_1_off]

  ; d_high_bit
  ; To get c, we shift right by d_high_bit
  movq mm4, [rbp + d_high_bit_off]

  ; d_mask
  ; To get d, we mask c_d_ with d_mask.
  mov rax, 1
  movq rcx, mm4
  shl rax, cl
  sub rax, 1
  movq mm5, rax
  
  ; step_table_c_d_ptr
  movq mm6, [rbp + step_table_c_d_off]
  
  ; tail_buf_ptr
  movq mm7, [rbp + step_table_exp3_off]
    
  ;
  ; Outer loop
  ;
  
.outer_loop:  	
  ; Set up the N that we will check in this round.
  movq n_0, mm0
  movq rax, mm1
  paddq mm1, [four]
  movsxd rbx, b32 [rax]
  add n_0, rbx
  mov n_1, 0
  ; Store N for use if we find a record for it.
  mov [current_n], n_0

  ; Start delay counter at zero.
  mov delay, 0

  ; We optimize the loop so that it checks only at the end. This would
  ; invalidate results if we started calculating at n_tmp < tail_size but we
  ; always start calculating above that.

  ; while (n_tmp >= tail_size) {

  ;
  ; Inner loop.
  ;
  
.inner_loop:
  ; Count rounds in loop for benchmarking. Hopefully, latency for these is
  ; hidden since we don't use the values.
  movq rax, mm2
  movq rbx, mm3
  add b64 [rax], 1
  adc b64 [rbx], 0

  ; Load c and d from step table.
  mov rax, n_0
  and rax, step_mask
  movq rbx, mm6
  movsxd c_d, b32 [rbx + rax * 4]

  ; u128 a(n_tmp >> step_bits)
  mov rcx, step_bits
  mov a_0, n_0
  mov a_1, n_1
  shrd a_0, a_1, cl
  shr a_1, cl

  ; n_tmp = a * (3 ^ c)
  mov rax, c_d
  movq rcx, mm4
  shr rax, cl
  movq rbx, mm7
  movsxd rbx, b32 [rbx + rax * 4]
  mov rax, a_0
  mul rbx       ; rdx:rax = rax * arg
  mov n_0, rax
  mov rcx, rdx
  mov rax, a_1
  mul rbx
  add rax, rcx
  mov n_1, rax

  ; n_tmp += d
  mov rax, c_d
  movq rbx, mm5
  and rax, rbx
  add n_0, rax
  adc n_1, 0

  ; delay += step_bits + c
  mov rax, c_d
  movq rcx, mm4
  shr rax, cl
  add rax, step_bits
  add delay, rax

  ; Keep iterating as long as N is higher than tail size.
  cmp n_1, 0
  jne .inner_loop
  cmp n_0, tail_size
  jae .inner_loop

  ;
  ; End inner loop.
  ;
  
  ; Look up the remaining delay from the tail table.
  ; delay += tail [n]
  mov rax, [rbp + tail_buf_off]
  movsxd rax, b32 [rax + n_0 * 4]
  add delay, rax

  ; if (delay > delay_high) {
  cmp delay, delay_high
  jg .new_record

  ; Check next N.
  sub b64 [sieve_size], 1; dec doesn't update CF.
  jne .outer_loop

  ; Return false for no new delay record.
  mov rax, 0
  jmp .end

  ;
  ; End outer loop.
  ;
  
.new_record:
  ; Found a new delay record. Store it and return true.
  mov rax, [current_n]
  mov rbx, [n_base_ptr]
  mov [rbx], rax
  mov rbx, [delay_high_ptr]
  mov [rbx], delay
  mov rax, 1

.end
  ; Restore the non-volatile registers.
  mov			rsi, [rbp + 0x00]
  mov			rdi, [rbp + 0x08]
  mov			rbx, [rbp + 0x10]
  mov			rcx, [rbp + 0x18]
  mov			r12, [rbp + 0x20]
  mov			r13, [rbp + 0x28]
  mov			r14, [rbp + 0x30]
  mov			r15, [rbp + 0x38]

  ; Official epilog
  lea			rsp, [rbp + 0x40]
  pop			rbp
  ret

ENDPROC_FRAME
