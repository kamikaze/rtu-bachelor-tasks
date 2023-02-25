output "ecr_image_dataset_converter_url" {
  value = var.ecr_enabled ? aws_ecr_repository.dataset_image_converter[0].repository_url : null
}

output "ecr_lambda_python_runtime_url" {
  value = var.ecr_enabled ? aws_ecr_repository.lambda_python_runtime[0].repository_url : null
}

output "k3s_master_ip" {
  value = var.ec2_enabled ? aws_spot_instance_request.k3s_master_spot_ec2_instance[0].public_ip : null
}

#output "rtu_cpu_monster_ip" {
#  value = var.ec2-enabled ? aws_spot_instance_request.rtu_cpu_spot_ec2_instance.public_ip : null
#}

#output "rtu_gpu_monster_ip" {
#  value = var.ec2-enabled ? aws_spot_instance_request.rtu_gpu_spot_ec2_instance.public_ip : null
#}

output "rtu_bachelor_db_master_ip" {
  value = var.rds_enabled ? aws_db_instance.rtu_bachelor_db_master[0].address : null
}

#output "rtu_bachelor_db_ro_replica_ip" {
#  value = var.rds-enabled ? aws_db_instance.rtu_bachelor_db_ro_replica[*].address : null
#}
