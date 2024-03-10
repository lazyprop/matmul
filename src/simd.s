    .text
    .global simd
    
simd:                                   # @simd
        pushq   %rbp
        movq    %rsp, %rbp
        andq    $-32, %rsp
        subq    $448, %rsp                      # imm = 0x1C0
        movq    %rdi, 216(%rsp)
        movq    %rsi, 208(%rsp)
        movq    %rdx, 200(%rsp)
        movl    $0, 196(%rsp)
.LBB0_1:                                # =>This Loop Header: Depth=1
        movl    196(%rsp), %eax
        cmpl    N(%rip), %eax
        jge     .LBB0_16
        movl    $0, 192(%rsp)
.LBB0_3:                                #   Parent Loop BB0_1 Depth=1
        movl    192(%rsp), %eax
        cmpl    N(%rip), %eax
        jge     .LBB0_14
        vxorps  %xmm0, %xmm0, %xmm0
        vmovaps %ymm0, 224(%rsp)
        vmovaps 224(%rsp), %ymm0
        vmovaps %ymm0, 160(%rsp)
        movl    $0, 156(%rsp)
.LBB0_5:                                #   Parent Loop BB0_1 Depth=1
        movl    156(%rsp), %eax
        cmpl    N(%rip), %eax
        jge     .LBB0_8
        movq    216(%rsp), %rax
        movl    196(%rsp), %ecx
        imull   N(%rip), %ecx
        addl    156(%rsp), %ecx
        movslq  %ecx, %rcx
        shlq    $2, %rcx
        addq    %rcx, %rax
        movq    %rax, 280(%rsp)
        movq    280(%rsp), %rax
        vmovups (%rax), %ymm0
        vmovaps %ymm0, 96(%rsp)
        movq    208(%rsp), %rax
        movl    192(%rsp), %ecx
        imull   N(%rip), %ecx
        addl    156(%rsp), %ecx
        movslq  %ecx, %rcx
        shlq    $2, %rcx
        addq    %rcx, %rax
        movq    %rax, 272(%rsp)
        movq    272(%rsp), %rax
        vmovups (%rax), %ymm0
        vmovaps %ymm0, 64(%rsp)
        vmovaps 96(%rsp), %ymm2
        vmovaps 64(%rsp), %ymm1
        vmovaps 160(%rsp), %ymm0
        vmovaps %ymm2, 352(%rsp)
        vmovaps %ymm1, 320(%rsp)
        vmovaps %ymm0, 288(%rsp)
        vmovaps 352(%rsp), %ymm1
        vmovaps 320(%rsp), %ymm0
        vmovaps 288(%rsp), %ymm2
        vfmadd213ps     %ymm2, %ymm1, %ymm0     # ymm0 = (ymm1 * ymm0) + ymm2
        vmovaps %ymm0, 160(%rsp)
        movl    156(%rsp), %eax
        addl    $8, %eax
        movl    %eax, 156(%rsp)
        jmp     .LBB0_5
.LBB0_8:                                #   in Loop: Header=BB0_3 Depth=2
        leaq    32(%rsp), %rax
        vmovaps 160(%rsp), %ymm0
        movq    %rax, 424(%rsp)
        vmovaps %ymm0, 384(%rsp)
        vmovaps 384(%rsp), %ymm0
        movq    424(%rsp), %rax
        vmovups %ymm0, (%rax)
        movl    $0, 28(%rsp)
.LBB0_9:                                #   Parent Loop BB0_1 Depth=1
        cmpl    $8, 28(%rsp)
        jge     .LBB0_12
        movslq  28(%rsp), %rax
        vmovss  32(%rsp,%rax,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
        movq    200(%rsp), %rax
        movl    196(%rsp), %ecx
        imull   N(%rip), %ecx
        addl    192(%rsp), %ecx
        movslq  %ecx, %rcx
        vaddss  (%rax,%rcx,4), %xmm0, %xmm0
        vmovss  %xmm0, (%rax,%rcx,4)
        movl    28(%rsp), %eax
        addl    $1, %eax
        movl    %eax, 28(%rsp)
        jmp     .LBB0_9
.LBB0_12:                               #   in Loop: Header=BB0_3 Depth=2
        jmp     .LBB0_13
.LBB0_13:                               #   in Loop: Header=BB0_3 Depth=2
        movl    192(%rsp), %eax
        addl    $1, %eax
        movl    %eax, 192(%rsp)
        jmp     .LBB0_3
.LBB0_14:                               #   in Loop: Header=BB0_1 Depth=1
        jmp     .LBB0_15
.LBB0_15:                               #   in Loop: Header=BB0_1 Depth=1
        movl    196(%rsp), %eax
        addl    $1, %eax
        movl    %eax, 196(%rsp)
        jmp     .LBB0_1
.LBB0_16:
        movq    %rbp, %rsp
        popq    %rbp
        vzeroupper
        retq
    .data
    N = 16
    
